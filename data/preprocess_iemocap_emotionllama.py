import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import av  # type: ignore
except Exception:
    av = None
import cv2
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from timm.models import create_model
from timm.models import hub as timm_hub
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel
import decord
decord.bridge.set_bridge('torch')

warnings.filterwarnings("ignore", category=UserWarning)


EMO_MAP_4 = {"ang", "hap", "exc", "sad", "neu"}
EMO_MAP_6 = {"ang", "hap", "exc", "sad", "neu", "fru"}


@dataclass
class Utterance:
    utt_id: str
    session: str
    dialog_id: str
    speaker: str
    start: float
    end: float
    label: str
    text: str


def resolve_iemocap_root(path: str) -> str:
    path = os.path.abspath(path)
    cands = [
        path,
        os.path.join(path, "IEMOCAP_full_release"),
    ]
    for cand in cands:
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "Session1")):
            return cand
    raise FileNotFoundError(
        f"Cannot locate IEMOCAP root from: {path}. "
        "Expected a folder containing Session1..Session5."
    )


def find_dialog_video(root: str, session: str, dialog_id: str) -> Optional[str]:
    cand1 = os.path.join(root, session, "dialog", "avi", "DivX", f"{dialog_id}.avi")
    cand2 = os.path.join(root, session, "dialog", "avi", f"{dialog_id}.avi")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


def find_dialog_wav(root: str, session: str, dialog_id: str) -> Optional[str]:
    cand = os.path.join(root, session, "dialog", "wav", f"{dialog_id}.wav")
    if os.path.isfile(cand):
        return cand
    return None


def parse_transcriptions(root: str) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    patt = re.compile(r"^(Ses\d{2}[FM]_[^ ]+_[FM]\d{3})\s+\[.*?\]:\s*(.*)$")
    for sid in range(1, 6):
        trans_dir = os.path.join(root, f"Session{sid}", "dialog", "transcriptions")
        if not os.path.isdir(trans_dir):
            continue
        for fn in os.listdir(trans_dir):
            if not fn.endswith(".txt"):
                continue
            fp = os.path.join(trans_dir, fn)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = patt.match(line.strip())
                    if m:
                        utt_id, text = m.group(1), m.group(2).strip()
                        texts[utt_id] = text
    return texts


def parse_emotions(root: str, keep_labels: set) -> Dict[str, Tuple[float, float, str, str, str]]:
    """
    Returns:
        utt_id -> (start, end, label, session, dialog_id)
    """
    info: Dict[str, Tuple[float, float, str, str, str]] = {}
    patt = re.compile(
        r"^\[(?P<s>\d+\.?\d*)\s*-\s*(?P<e>\d+\.?\d*)\]\t(?P<uid>Ses\d{2}[FM]_[^\t]+)\t(?P<emo>[a-zA-Z]+)"
    )
    for sid in range(1, 6):
        session = f"Session{sid}"
        emo_dir = os.path.join(root, session, "dialog", "EmoEvaluation")
        if not os.path.isdir(emo_dir):
            continue
        for fn in os.listdir(emo_dir):
            if not fn.endswith(".txt"):
                continue
            dialog_id = os.path.splitext(fn)[0]
            fp = os.path.join(emo_dir, fn)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("["):
                        continue
                    m = patt.match(line)
                    if not m:
                        continue
                    start = float(m.group("s"))
                    end = float(m.group("e"))
                    utt_id = m.group("uid")
                    emo = m.group("emo").lower()
                    if emo not in keep_labels:
                        continue
                    info[utt_id] = (start, end, emo, session, dialog_id)
    return info


def build_utterances(root: str, classes: int) -> List[Utterance]:
    keep = EMO_MAP_4 if classes == 4 else EMO_MAP_6
    emo = parse_emotions(root, keep)
    txt = parse_transcriptions(root)
    utts: List[Utterance] = []
    for utt_id, (start, end, label, session, dialog_id) in emo.items():
        tail = utt_id.rsplit("_", 1)[-1] if "_" in utt_id else ""
        speaker = "speaker_1" if tail.startswith("M") else "speaker_2"
        text = txt.get(utt_id, "")
        utts.append(
            Utterance(
                utt_id=utt_id,
                session=session,
                dialog_id=dialog_id,
                speaker=speaker,
                start=start,
                end=end,
                label=label,
                text=text,
            )
        )
    utts.sort(key=lambda x: (x.session, x.dialog_id, x.start, x.utt_id))
    return utts


def adaptive_downsample_with_padding(features: torch.Tensor, target_len: int) -> torch.Tensor:
    # features: [1, T, D]
    if features.shape[1] < target_len:
        features = F.pad(features, (0, 0, 0, target_len - features.shape[1]))
    features = features.transpose(1, 2)  # [1, D, T]
    features = F.adaptive_avg_pool1d(features, target_len)
    return features.transpose(1, 2)  # [1, target_len, D]


def spatiotemporal_downsample(x: torch.Tensor, target_h: int, target_w: int, target_t: int) -> torch.Tensor:
    # x: [B, T, H, W, C]
    x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
    x = F.adaptive_avg_pool3d(x, (x.shape[2], target_h, target_w))
    if x.shape[2] != target_t:
        x = F.adaptive_avg_pool3d(x, (target_t, target_h, target_w))
    x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
    b, t, h, w, c = x.shape
    return x.reshape(b, t * h * w, c)  # [B, T*H*W, C]


def sample_frame_indices(clip_len: int, frame_sample_rate: int, seg_len: int) -> np.ndarray:
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= 0:
        return np.zeros((clip_len,), dtype=np.int64)
    if seg_len <= converted_len:
        return np.clip(np.linspace(0, seg_len - 1, num=clip_len), 0, max(seg_len - 1, 0)).astype(np.int64)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    return np.clip(np.linspace(start_idx, end_idx, num=clip_len), start_idx, end_idx - 1).astype(np.int64)


def read_video_cv2_abs(file_path: str, indices_abs: np.ndarray, clip_len: int = 16) -> List[np.ndarray]:
    # 替换为基于 decord 的极速读取方案
    try:
        vr = decord.VideoReader(file_path, ctx=decord.cpu(0))
        # decord 支持直接传入数组，一次性在底层解码所需的帧
        frames_tensor = vr.get_batch(indices_abs).numpy() 
        frames = [f for f in frames_tensor]
    except Exception as e:
        print(f"Decord failed to read {file_path}, fallback to empty frames. Error: {e}")
        frames = []

    if len(frames) == 0:
        raise ValueError("No frames read.")
    while len(frames) < clip_len:
        frames.append(frames[-1].copy())
    return frames


def read_video_pyav_abs(file_path: str, indices_abs: np.ndarray, clip_len: int = 16) -> List[np.ndarray]:
    if av is None:
        raise ImportError("pyav is not installed.")
    container = av.open(file_path)
    frames = []
    container.seek(0)
    idx_set = set(int(x) for x in indices_abs.tolist())
    start_index = int(indices_abs[0])
    end_index = int(indices_abs[-1])
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in idx_set:
            frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if len(frames) == 0:
        raise ValueError("No frames read with pyav.")
    while len(frames) < clip_len:
        frames.append(frames[-1].copy())
    return frames


class EmotionLLaMAStyleExtractor:
    """
    Extracts Whisper-large-v3 and EVA-ViT-G features in Emotion-LLaMA-v2 style,
    with two output modes:
      - raw:
          audio [64, 1280], video [64, 1408]
      - compressed:
          audio [64, 1280] -> dim-compress [64, 64] -> time-resample [157, 64]
          video [64, 1408] -> dim-compress [64, 64] -> time-resample [32, 64]
    """

    def __init__(
        self,
        whisper_model: str,
        eva_ckpt: Optional[str],
        device: str,
        mode: str = "raw",
        amp: bool = True,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.whisper_model_name = whisper_model
        self.eva_ckpt = eva_ckpt
        self.mode = mode.lower()
        if self.mode not in {"raw", "compressed"}:
            raise ValueError(f"Unsupported mode: {mode}. Choose from raw/compressed.")
        self.use_amp = bool(amp and self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self._whisper = None
        self._whisper_fe = None
        self._eva = None
        self._eva_tf = None

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def _load_whisper(self):
        if self._whisper is None:
            whisper_dtype = torch.float16 if self.device.type == "cuda" else None
            self._whisper = WhisperModel.from_pretrained(
                self.whisper_model_name,
                torch_dtype=whisper_dtype,
            ).to(self.device)
            self._whisper.eval()
            self._whisper_fe = AutoFeatureExtractor.from_pretrained(self.whisper_model_name)

    def _load_eva(self):
        if self._eva is not None:
            return
        self._eva = create_model("eva_giant_patch14_224", pretrained=False, num_classes=0, global_pool="")
        ckpt = self.eva_ckpt
        if ckpt is None:
            ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/eva_vit_g.pth")
            if not os.path.exists(ckpt):
                url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
                ckpt = timm_hub.download_cached_file(url, check_hash=False, progress=True)
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self._eva.load_state_dict(state, strict=False)
        self._eva = self._eva.to(self.device)
        self._eva.eval()
        self._eva_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _compress_dim_block_mean(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        # x: [T, D], D must be divisible by out_dim
        t, d = x.shape
        if d % out_dim != 0:
            raise ValueError(f"Feature dim {d} cannot be block-compressed to {out_dim}.")
        blk = d // out_dim
        return x.view(t, out_dim, blk).mean(dim=-1)

    @staticmethod
    def _resample_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        # x: [T, D] -> [target_t, D]
        x = x.transpose(0, 1).unsqueeze(0)  # [1, D, T]
        x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1).contiguous()

    def extract_audio_from_wave_segment(self, wave: torch.Tensor, sr: int) -> np.ndarray:
        self._load_whisper()
        if wave.ndim == 2:
            wave = wave.mean(dim=0)
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        wave_np = wave.detach().cpu().numpy()
        inputs = self._whisper_fe(wave_np, sampling_rate=16000, return_tensors="pt")
        # input_features = inputs.input_features.to(self.device)
        input_features = inputs.input_features.to(
            device=self.device,
            dtype=next(self._whisper.parameters()).dtype
        )
        decoder_input_ids = torch.tensor([[self._whisper.config.decoder_start_token_id]], device=self.device)
        with torch.no_grad():
            out = self._whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
        feat = out.encoder_last_hidden_state  # [1, T, 1280]
        feat = adaptive_downsample_with_padding(feat, target_len=64).squeeze(0)  # [64, 1280]

        if self.mode == "raw":
            return feat.float().cpu().numpy().astype(np.float32)

        feat = self._compress_dim_block_mean(feat, out_dim=64)  # [64, 64]
        feat = self._resample_time(feat, target_t=157)  # [157, 64]
        return feat.float().cpu().numpy().astype(np.float32)

    def extract_video_from_segment(self, video_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        self._load_eva()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 1e-6 or total_frames <= 0:
            if self.mode == "raw":
                return np.zeros((64, 1408), dtype=np.float32)
            return np.zeros((32, 64), dtype=np.float32)

        s = max(0, int(np.floor(start_sec * fps)))
        e = min(total_frames - 1, int(np.ceil(end_sec * fps)))
        seg_len = max(1, e - s + 1)
        rel_idx = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)
        abs_idx = (s + rel_idx).astype(np.int64)

        try:
            frames = read_video_cv2_abs(video_path, abs_idx, clip_len=16)
        except Exception:
            frames = read_video_pyav_abs(video_path, abs_idx, clip_len=16)

        images_tensor = torch.stack([self._eva_tf(Image.fromarray(f)) for f in frames], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode():
            with self._autocast_ctx():
                feat = self._eva.forward_features(images_tensor)  # [16, 257, 1408]
        feat = feat[:, 1:, :].reshape(1, 16, 16, 16, -1)  # [1,16,16,16,1408]
        feat = spatiotemporal_downsample(feat, 2, 2, 16).squeeze(0)  # [64,1408]

        if self.mode == "raw":
            return feat.float().cpu().numpy().astype(np.float32)

        feat = self._compress_dim_block_mean(feat, out_dim=64)  # [64,64]
        feat = self._resample_time(feat, target_t=32)  # [32,64]
        return feat.float().cpu().numpy().astype(np.float32)


def stable_valid_split(utt_id: str, valid_ratio: float, seed: int) -> bool:
    key = f"{utt_id}-{seed}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16) % 1_000_000
    return h < int(valid_ratio * 1_000_000)


def speaker_tag_map(utts: List[Utterance]) -> Dict[str, str]:
    speakers = []
    for u in utts:
        suffix = u.utt_id.split("_")[-1][0]
        if suffix not in speakers:
            speakers.append(suffix)
    mapping = {}
    if len(speakers) >= 1:
        mapping[speakers[0]] = "a"
    if len(speakers) >= 2:
        mapping[speakers[1]] = "b"
    return mapping


def build_context_for_dialog(dialog_utts: List[Utterance]) -> Dict[str, str]:
    sp_map = speaker_tag_map(dialog_utts)
    seq = []
    for u in dialog_utts:
        s = u.utt_id.split("_")[-1][0]
        tag = sp_map.get(s, "a")
        seq.append(f"<{tag}>{u.text}")
    ctx_json = json.dumps(seq, ensure_ascii=False)
    return {u.utt_id: ctx_json for u in dialog_utts}


def write_split_csv(path: str, rows: List[Dict[str, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["text", "context", "speaker", "label", "index", "vid_cid"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(rows):
            out = dict(r)
            out["index"] = i
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IEMOCAP full release into MMER-Adapter format using Emotion-LLaMA-v2 extraction style."
    )
    parser.add_argument(
        "--iemocap_root",
        type=str,
        default="datasets/IEMOCAP_full_release",
        help="Path to IEMOCAP_full_release or its parent directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/IEMOCAP",
        help="Output directory, will contain iemocap_data_0610.pkl and iemocap_text/*.csv.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-large-v3",
        help="HuggingFace path or local path for Whisper-large-v3.",
    )
    parser.add_argument(
        "--eva_ckpt",
        type=str,
        default=None,
        help="Path to eva_vit_g.pth. If omitted, uses ~/.cache or downloads automatically.",
    )
    parser.add_argument("--classes", type=int, default=6, choices=[4, 6], help="IEMOCAP label set size.")
    parser.add_argument("--valid_ratio", type=float, default=0.09, help="Validation ratio inside Session1-4.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP on CUDA (use --no-amp to disable).",
    )
    parser.add_argument("--max_utts", type=int, default=None, help="Debug limit for number of utterances.")
    parser.add_argument(
        "--mode",
        type=str,
        default="raw",
        choices=["raw", "compressed"],
        help=(
            "Feature output mode. "
            "raw: audio[64,1280], video[64,1408]; "
            "compressed: audio[157,64], video[32,64]."
        ),
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    iemocap_root = resolve_iemocap_root(args.iemocap_root)
    output_dir = os.path.abspath(args.output_dir)
    text_dir = os.path.join(output_dir, "iemocap_text")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    print(f"[Info] IEMOCAP root: {iemocap_root}")
    print(f"[Info] Output dir:   {output_dir}")

    utterances = build_utterances(iemocap_root, args.classes)
    if args.max_utts is not None:
        utterances = utterances[: args.max_utts]
    print(f"[Info] Parsed utterances: {len(utterances)}")
    if len(utterances) == 0:
        raise RuntimeError("No utterances found after parsing/filtering.")

    # Group by dialog for context + efficient audio loading
    dialog_groups: Dict[Tuple[str, str], List[Utterance]] = defaultdict(list)
    for u in utterances:
        dialog_groups[(u.session, u.dialog_id)].append(u)
    for k in dialog_groups:
        dialog_groups[k].sort(key=lambda x: (x.start, x.utt_id))

    # Build context map
    context_map: Dict[str, str] = {}
    for _, group in dialog_groups.items():
        context_map.update(build_context_for_dialog(group))

    # Prepare split containers
    train_rows, valid_rows, test_rows = [], [], []
    audio_splits = [dict(), dict(), dict()]  # train, valid, test
    video_splits = [dict(), dict(), dict()]

    extractor = EmotionLLaMAStyleExtractor(
        whisper_model=args.whisper_model,
        eva_ckpt=args.eva_ckpt,
        device=args.device,
        mode=args.mode,
        amp=args.amp,
    )

    pbar = tqdm(total=len(utterances), desc="Extracting features", ncols=120)
    for (session, dialog_id), group in dialog_groups.items():
        wav_path = find_dialog_wav(iemocap_root, session, dialog_id)
        video_path = find_dialog_video(iemocap_root, session, dialog_id)
        if wav_path is None or video_path is None:
            pbar.update(len(group))
            continue

        try:
            wav, sr = torchaudio.load(wav_path)
        except Exception:
            pbar.update(len(group))
            continue

        for u in group:
            start_sample = max(0, int(u.start * sr))
            end_sample = min(wav.shape[1], int(u.end * sr))
            if end_sample <= start_sample:
                pbar.update(1)
                continue
            wav_seg = wav[:, start_sample:end_sample]

            try:
                a_feat = extractor.extract_audio_from_wave_segment(wav_seg, sr)
                v_feat = extractor.extract_video_from_segment(video_path, u.start, u.end)
            except Exception as e:
                print(f"[Skip] {u.utt_id}: {type(e).__name__}: {e}")
                pbar.update(1)
                continue

            if u.session == "Session5":
                split_idx = 2
                row_list = test_rows
            else:
                is_valid = stable_valid_split(u.utt_id, args.valid_ratio, args.seed)
                split_idx = 1 if is_valid else 0
                row_list = valid_rows if is_valid else train_rows

            audio_splits[split_idx][u.utt_id] = a_feat
            video_splits[split_idx][u.utt_id] = v_feat
            row_list.append(
                {
                    "text": u.text,
                    "context": context_map.get(u.utt_id, "[]"),
                    "speaker": u.speaker,
                    "label": u.label,
                    "vid_cid": u.utt_id,
                }
            )
            pbar.update(1)

    pbar.close()

    # Save pkl in MMER expected format
    out_pkl = os.path.join(output_dir, "iemocap_data_0610.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"audio": audio_splits, "video": video_splits}, f, protocol=4)

    # Save split CSVs for MMER load_data.py
    write_split_csv(os.path.join(text_dir, "iemocap_data_train.csv"), train_rows)
    write_split_csv(os.path.join(text_dir, "iemocap_data_valid.csv"), valid_rows)
    write_split_csv(os.path.join(text_dir, "iemocap_data_test.csv"), test_rows)

    stats = {
        "mode": args.mode,
        "classes": args.classes,
        "num_train": len(train_rows),
        "num_valid": len(valid_rows),
        "num_test": len(test_rows),
        "pkl_path": out_pkl,
        "csv_dir": text_dir,
        "audio_shape": [64, 1280] if args.mode == "raw" else [157, 64],
        "video_shape": [64, 1408] if args.mode == "raw" else [32, 64],
    }
    with open(os.path.join(output_dir, "preprocess_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[Done] Preprocessing complete.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(
        "\n[Next] Example training command:\n"
        f"python MMER-Adapter/run.py --datasetName iemocap6 --iemocap_feature_mode {args.mode} "
        "--root_dataset_dir datasets --model_type chatglm3 --pretrain_LM /path/to/your/llm"
    )


if __name__ == "__main__":
    main()

import argparse
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
import timm.models.hub as timm_hub
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel

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
    cands = [path, os.path.join(path, "IEMOCAP_full_release")]
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
    if os.path.isfile(cand1): return cand1
    if os.path.isfile(cand2): return cand2
    return None

def find_dialog_wav(root: str, session: str, dialog_id: str) -> Optional[str]:
    cand = os.path.join(root, session, "dialog", "wav", f"{dialog_id}.wav")
    if os.path.isfile(cand): return cand
    return None

def parse_transcriptions(root: str) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    patt = re.compile(r"^(Ses\d{2}[FM]_[^ ]+_[FM]\d{3})\s+\[.*?\]:\s*(.*)$")
    for sid in range(1, 6):
        trans_dir = os.path.join(root, f"Session{sid}", "dialog", "transcriptions")
        if not os.path.isdir(trans_dir): continue
        for fn in os.listdir(trans_dir):
            if not fn.endswith(".txt"): continue
            fp = os.path.join(trans_dir, fn)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = patt.match(line.strip())
                    if m: texts[m.group(1)] = m.group(2).strip()
    return texts

def parse_emotions(root: str, keep_labels: set) -> Dict[str, Tuple[float, float, str, str, str]]:
    info: Dict[str, Tuple[float, float, str, str, str]] = {}
    patt = re.compile(r"^\[(?P<s>\d+\.?\d*)\s*-\s*(?P<e>\d+\.?\d*)\]\t(?P<uid>Ses\d{2}[FM]_[^\t]+)\t(?P<emo>[a-zA-Z]+)")
    for sid in range(1, 6):
        session = f"Session{sid}"
        emo_dir = os.path.join(root, session, "dialog", "EmoEvaluation")
        if not os.path.isdir(emo_dir): continue
        for fn in os.listdir(emo_dir):
            if not fn.endswith(".txt"): continue
            dialog_id = os.path.splitext(fn)[0]
            fp = os.path.join(emo_dir, fn)
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("["): continue
                    m = patt.match(line)
                    if not m: continue
                    start, end = float(m.group("s")), float(m.group("e"))
                    emo = m.group("emo").lower()
                    if emo not in keep_labels: continue
                    info[m.group("uid")] = (start, end, emo, session, dialog_id)
    return info

def build_utterances(root: str, classes: int) -> List[Utterance]:
    keep = EMO_MAP_4 if classes == 4 else EMO_MAP_6
    emo = parse_emotions(root, keep)
    txt = parse_transcriptions(root)
    utts: List[Utterance] = []
    for utt_id, (start, end, label, session, dialog_id) in emo.items():
        tail = utt_id.rsplit("_", 1)[-1] if "_" in utt_id else ""
        speaker = "speaker_1" if tail.startswith("M") else "speaker_2"
        utts.append(Utterance(utt_id, session, dialog_id, speaker, start, end, label, txt.get(utt_id, "")))
    utts.sort(key=lambda x: (x.session, x.dialog_id, x.start, x.utt_id))
    return utts

def adaptive_downsample_with_padding(features: torch.Tensor, target_len: int) -> torch.Tensor:
    if features.shape[1] < target_len:
        features = F.pad(features, (0, 0, 0, target_len - features.shape[1]))
    features = features.transpose(1, 2) 
    features = F.adaptive_avg_pool1d(features, target_len)
    return features.transpose(1, 2) 

def spatiotemporal_downsample(x: torch.Tensor, target_h: int, target_w: int, target_t: int) -> torch.Tensor:
    x = x.permute(0, 4, 1, 2, 3) 
    x = F.adaptive_avg_pool3d(x, (x.shape[2], target_h, target_w))
    if x.shape[2] != target_t:
        x = F.adaptive_avg_pool3d(x, (target_t, target_h, target_w))
    x = x.permute(0, 2, 3, 4, 1) 
    b, t, h, w, c = x.shape
    return x.reshape(b, t * h * w, c) 

def sample_frame_indices(clip_len: int, frame_sample_rate: int, seg_len: int) -> np.ndarray:
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= 0: return np.zeros((clip_len,), dtype=np.int64)
    if seg_len <= converted_len:
        return np.clip(np.linspace(0, seg_len - 1, num=clip_len), 0, max(seg_len - 1, 0)).astype(np.int64)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    return np.clip(np.linspace(start_idx, end_idx, num=clip_len), start_idx, end_idx - 1).astype(np.int64)

# 降级备用读取方案（当 decord 遇到损坏的 AVI 时调用）
def read_video_cv2_abs(file_path: str, indices_abs: np.ndarray, clip_len: int = 16) -> List[np.ndarray]:
    cap = cv2.VideoCapture(file_path)
    frames: List[np.ndarray] = []
    if cap.isOpened():
        for idx in indices_abs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    if len(frames) == 0:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    while len(frames) < clip_len: frames.append(frames[-1].copy())
    return frames

class EmotionLLaMAStyleExtractor:
    """Always extracts raw features; compression is applied as post-processing."""
    def __init__(self, whisper_model: str, eva_ckpt: Optional[str], device: str):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.whisper_model_name = whisper_model
        self.eva_ckpt = eva_ckpt
        self._whisper, self._whisper_fe = None, None
        self._eva, self._eva_tf = None, None

    def _load_whisper(self):
        if self._whisper is None:
            print("[Model] Loading Whisper...")
            self._whisper = WhisperModel.from_pretrained(self.whisper_model_name).to(self.device)
            self._whisper.eval()
            self._whisper_fe = AutoFeatureExtractor.from_pretrained(self.whisper_model_name)

    def _unload_whisper(self):
        """Free Whisper from VRAM to make room for EVA-ViT."""
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
            torch.cuda.empty_cache()
            print("[Model] Whisper unloaded, VRAM freed.")

    def _load_eva(self):
        if self._eva is not None: return
        print("[Model] Loading EVA-ViT-G...")
        self._eva = create_model("eva_giant_patch14_224", pretrained=False, num_classes=0, global_pool="")
        ckpt = self.eva_ckpt or os.path.expanduser("~/.cache/torch/hub/checkpoints/eva_vit_g.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state: state = state["model"]
        self._eva.load_state_dict(state, strict=False)
        self._eva = self._eva.to(self.device)
        self._eva.eval()
        self._eva_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _unload_eva(self):
        """Free EVA-ViT from VRAM."""
        if self._eva is not None:
            del self._eva
            self._eva = None
            torch.cuda.empty_cache()
            print("[Model] EVA-ViT unloaded, VRAM freed.")

    # NOTE: static compression methods below are kept for reference.
    # Actual compression is done by module-level compress_features() after extraction.

    @staticmethod
    def _compress_dim_block_mean(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        """Legacy: block-mean compression (not recommended, see analysis)."""
        t, d = x.shape
        blk = d // out_dim
        return x.view(t, out_dim, blk).mean(dim=-1)

    @staticmethod
    def _compress_dim_head_aware(x: torch.Tensor, out_dim: int, num_heads: int) -> torch.Tensor:
        """Head-aware compression: pool within each attention head independently.

        Avoids mixing information across different attention heads, preserving
        the multi-head structure of Whisper/EVA-ViT encoder outputs.

        When out_dim is not evenly divisible by num_heads (e.g. 64/20),
        the first `remainder` heads get one extra output dim.
        """
        t, d = x.shape
        head_dim = d // num_heads
        base_out = out_dim // num_heads
        remainder = out_dim % num_heads

        x_heads = x.view(t, num_heads, head_dim)  # (T, H, head_dim)
        compressed = []
        for h in range(num_heads):
            target = base_out + (1 if h < remainder else 0)
            pooled = F.adaptive_avg_pool1d(
                x_heads[:, h, :].unsqueeze(1), target  # (T, 1, head_dim) -> (T, 1, target)
            ).squeeze(1)  # (T, target)
            compressed.append(pooled)
        return torch.cat(compressed, dim=-1)  # (T, out_dim)

    @staticmethod
    def _resample_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0) 
        x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1).contiguous()

    # ====== 新增：批量处理核心大招 ======
    def extract_audio_batch(self, wave: torch.Tensor, sr: int, time_segments: List[Tuple[float, float]], batch_size: int = 16) -> List[np.ndarray]:
        self._load_whisper()
        if wave.ndim == 2: wave = wave.mean(dim=0)
        if sr != 16000: wave = torchaudio.functional.resample(wave, sr, 16000)
        wave_np = wave.detach().cpu().numpy()
        
        segments = []
        for start_sec, end_sec in time_segments:
            s_idx = max(0, int(start_sec * 16000))
            e_idx = min(len(wave_np), int(end_sec * 16000))
            segments.append(wave_np[s_idx:e_idx])
            
        results = []
        for i in range(0, len(segments), batch_size):
            batch_segs = segments[i : i+batch_size]
            actual_b = len(batch_segs)
            
            # inputs = self._whisper_fe(batch_segs, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = self._whisper_fe(batch_segs, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = inputs.input_features.to(device=self.device, dtype=next(self._whisper.parameters()).dtype)
            decoder_input_ids = torch.tensor([[self._whisper.config.decoder_start_token_id]] * actual_b, device=self.device)
            
            with torch.no_grad():
                out = self._whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
            feat = out.encoder_last_hidden_state 
            
            for j in range(actual_b):
                f = feat[j:j+1]
                f = adaptive_downsample_with_padding(f, target_len=64).squeeze(0)
                results.append(f.float().cpu().numpy().astype(np.float32))  # always raw: (64, 1280)
        return results

    def extract_video_batch(self, video_path: str, time_segments: List[Tuple[float, float]], batch_size: int = 8) -> List[np.ndarray]:
        self._load_eva()
        import decord
        decord.bridge.set_bridge('torch')
        
        use_decord = True
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
        except Exception:
            # 智能降级：如果遇到破损的老AVI，退回 cv2 模式获取元数据
            use_decord = False
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        all_images_tensors = []
        for start_sec, end_sec in time_segments:
            s = max(0, int(np.floor(start_sec * fps)))
            e = min(total_frames - 1, int(np.ceil(end_sec * fps)))
            seg_len = max(1, e - s + 1)
            rel_idx = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)
            abs_idx = (s + rel_idx).astype(np.int64)
            
            if use_decord:
                try:
                    frames = vr.get_batch(abs_idx).numpy()
                except Exception:
                    frames = read_video_cv2_abs(video_path, abs_idx)
            else:
                frames = read_video_cv2_abs(video_path, abs_idx)
                
            images_tensor = torch.stack([self._eva_tf(Image.fromarray(f)) for f in frames], dim=0)
            all_images_tensors.append(images_tensor)
            
        results = []
        for i in range(0, len(all_images_tensors), batch_size):
            batch_list = all_images_tensors[i : i+batch_size]
            actual_b = len(batch_list)
            
            batch_tensor = torch.stack(batch_list, dim=0).view(actual_b * 16, 3, 224, 224).to(self.device)
            
            with torch.no_grad():
                feat = self._eva.forward_features(batch_tensor)
                
            feat = feat.view(actual_b, 16, 257, 1408)
            feat = feat[:, :, 1:, :].reshape(actual_b, 16, 16, 16, -1)
            feat = spatiotemporal_downsample(feat, 2, 2, 16)
            
            for j in range(actual_b):
                f = feat[j]
                results.append(f.float().cpu().numpy().astype(np.float32))  # always raw: (64, 1408)
        return results

# ══════════════════════════════════════════════════════════════
# Post-extraction Compression (applied when --mode compressed)
# ══════════════════════════════════════════════════════════════
WHISPER_NUM_HEADS = 20   # whisper-large-v3: 1280 / 20 = 64 per head
EVA_VIT_NUM_HEADS = 16   # EVA-ViT-G:       1408 / 16 = 88 per head

def _resample_time_np(x: np.ndarray, target_t: int) -> np.ndarray:
    """Resample time dimension via linear interpolation. (T, D) -> (target_t, D)."""
    xt = torch.from_numpy(x).float().transpose(0, 1).unsqueeze(0)
    xt = F.interpolate(xt, size=target_t, mode="linear", align_corners=False)
    return xt.squeeze(0).transpose(0, 1).contiguous().numpy().astype(np.float32)

def _compress_block_mean_np(x: np.ndarray, out_dim: int) -> np.ndarray:
    """Legacy block-mean: split feature dim into blocks and average."""
    t, d = x.shape
    blk = d // out_dim
    return x.reshape(t, out_dim, blk).mean(axis=-1).astype(np.float32)

def _compress_head_aware_np(x: np.ndarray, out_dim: int, num_heads: int) -> np.ndarray:
    """Head-aware compression: pool within each attention head independently.
    
    Distributes output dims across heads. If out_dim is not evenly divisible
    by num_heads, the first `remainder` heads get one extra dim.
    E.g. Whisper: 64 / 20 heads -> first 4 heads get 4 dims, rest get 3.
    """
    t, d = x.shape
    head_dim = d // num_heads
    base_out = out_dim // num_heads
    remainder = out_dim % num_heads

    xt = torch.from_numpy(x).float().view(t, num_heads, head_dim)
    compressed = []
    for h in range(num_heads):
        target = base_out + (1 if h < remainder else 0)
        pooled = F.adaptive_avg_pool1d(
            xt[:, h, :].unsqueeze(1), target
        ).squeeze(1)
        compressed.append(pooled)
    return torch.cat(compressed, dim=-1).numpy().astype(np.float32)

def _apply_per_sample(splits, compress_fn, target_t):
    """Apply a per-sample compress function + time resampling to all splits."""
    for split_dict in splits:
        for uid in split_dict:
            f = compress_fn(split_dict[uid])
            split_dict[uid] = _resample_time_np(f, target_t)

def compress_features(audio_splits, video_splits, method, out_dim=64,
                      audio_target_t=157, video_target_t=32):
    """Compress raw features to low-dim format for backward compatibility.

    Methods:
        pca:        PCA fitted on entire dataset (best information retention)
        head_aware: Pool within each attention head (preserves multi-head structure)
        block_mean: Legacy block-mean (not recommended)
    """
    print(f"\n[Compress] method={method}, out_dim={out_dim}, "
          f"audio_T={audio_target_t}, video_T={video_target_t}")

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError(
                "PCA compression requires scikit-learn: pip install scikit-learn")

        # ── Audio PCA: fit on all frames across all splits ──
        all_audio = np.vstack([f for d in audio_splits for f in d.values()])
        pca_a = PCA(n_components=out_dim)
        pca_a.fit(all_audio)
        var_a = pca_a.explained_variance_ratio_.sum()
        print(f"[PCA Audio] {all_audio.shape[1]}d -> {out_dim}d, "
              f"explained variance: {var_a:.4f} ({var_a*100:.1f}%)")
        for split_dict in audio_splits:
            for uid in split_dict:
                f = pca_a.transform(split_dict[uid]).astype(np.float32)
                split_dict[uid] = _resample_time_np(f, audio_target_t)
        del all_audio

        # ── Video PCA ──
        all_video = np.vstack([f for d in video_splits for f in d.values()])
        pca_v = PCA(n_components=out_dim)
        pca_v.fit(all_video)
        var_v = pca_v.explained_variance_ratio_.sum()
        print(f"[PCA Video] {all_video.shape[1]}d -> {out_dim}d, "
              f"explained variance: {var_v:.4f} ({var_v*100:.1f}%)")
        for split_dict in video_splits:
            for uid in split_dict:
                f = pca_v.transform(split_dict[uid]).astype(np.float32)
                split_dict[uid] = _resample_time_np(f, video_target_t)
        del all_video

    elif method == "head_aware":
        print(f"[Head-Aware] Audio heads={WHISPER_NUM_HEADS}, "
              f"Video heads={EVA_VIT_NUM_HEADS}")
        _apply_per_sample(
            audio_splits,
            lambda f: _compress_head_aware_np(f, out_dim, WHISPER_NUM_HEADS),
            audio_target_t,
        )
        _apply_per_sample(
            video_splits,
            lambda f: _compress_head_aware_np(f, out_dim, EVA_VIT_NUM_HEADS),
            video_target_t,
        )

    elif method == "block_mean":
        print("[Block-Mean] WARNING: legacy method, not recommended")
        _apply_per_sample(
            audio_splits,
            lambda f: _compress_block_mean_np(f, out_dim),
            audio_target_t,
        )
        _apply_per_sample(
            video_splits,
            lambda f: _compress_block_mean_np(f, out_dim),
            video_target_t,
        )

    print("[Compress] Done.\n")
    return audio_splits, video_splits


def stable_valid_split(utt_id: str, valid_ratio: float, seed: int) -> bool:
    key = f"{utt_id}-{seed}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16) % 1_000_000
    return h < int(valid_ratio * 1_000_000)

def speaker_tag_map(utts: List[Utterance]) -> Dict[str, str]:
    speakers = []
    for u in utts:
        suffix = u.utt_id.split("_")[-1][0]
        if suffix not in speakers: speakers.append(suffix)
    mapping = {}
    if len(speakers) >= 1: mapping[speakers[0]] = "a"
    if len(speakers) >= 2: mapping[speakers[1]] = "b"
    return mapping

def build_context_for_dialog(dialog_utts: List[Utterance]) -> Dict[str, str]:
    sp_map = speaker_tag_map(dialog_utts)
    seq = [f"<{sp_map.get(u.utt_id.split('_')[-1][0], 'a')}>{u.text}" for u in dialog_utts]
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
    parser = argparse.ArgumentParser(description="Preprocess IEMOCAP format using Batching.")
    parser.add_argument("--iemocap_root", type=str, default="datasets/IEMOCAP_full_release")
    parser.add_argument("--output_dir", type=str, default="datasets/IEMOCAP")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--eva_ckpt", type=str, default=None)
    parser.add_argument("--classes", type=int, default=6, choices=[4, 6])
    parser.add_argument("--valid_ratio", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_utts", type=int, default=None)
    parser.add_argument("--mode", type=str, default="raw", choices=["raw", "compressed"])
    parser.add_argument("--compress_method", type=str, default="pca",
                        choices=["pca", "head_aware", "block_mean"],
                        help="Dimension compression method for compressed mode (default: pca)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    iemocap_root = resolve_iemocap_root(args.iemocap_root)
    output_dir = os.path.abspath(args.output_dir)
    text_dir = os.path.join(output_dir, "iemocap_text")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    print(f"[Info] IEMOCAP root: {iemocap_root}")
    print(f"[Info] Output dir:   {output_dir}")

    utterances = build_utterances(iemocap_root, args.classes)
    if args.max_utts is not None: utterances = utterances[: args.max_utts]
    print(f"[Info] Parsed utterances: {len(utterances)}")

    dialog_groups: Dict[Tuple[str, str], List[Utterance]] = defaultdict(list)
    for u in utterances: dialog_groups[(u.session, u.dialog_id)].append(u)
    for k in dialog_groups: dialog_groups[k].sort(key=lambda x: (x.start, x.utt_id))

    context_map: Dict[str, str] = {}
    for _, group in dialog_groups.items(): context_map.update(build_context_for_dialog(group))

    train_rows, valid_rows, test_rows = [], [], []
    audio_splits, video_splits = [dict(), dict(), dict()], [dict(), dict(), dict()]

    extractor = EmotionLLaMAStyleExtractor(args.whisper_model, args.eva_ckpt, args.device)

    # ══════════════════════════════════════════════════════
    # Two-pass extraction: load only ONE large model at a time
    # to avoid CUDA OOM on 24GB GPUs (Whisper ~3GB + EVA ~4.5GB)
    # ══════════════════════════════════════════════════════

    # Pre-build dialog metadata (wav/video paths, time segments, split indices)
    dialog_meta = []  # list of (key, group, wav_path, video_path, time_segments)
    for (session, dialog_id), group in dialog_groups.items():
        wav_path = find_dialog_wav(iemocap_root, session, dialog_id)
        video_path = find_dialog_video(iemocap_root, session, dialog_id)
        if not wav_path or not video_path:
            continue
        time_segments = [(u.start, u.end) for u in group]
        dialog_meta.append(((session, dialog_id), group, wav_path, video_path, time_segments))

    # ── Pass 1: Audio (Whisper only in VRAM) ──
    print(f"\n{'='*60}")
    print(f"Pass 1/2: Extracting audio features (Whisper)")
    print(f"{'='*60}")
    audio_map: Dict[str, np.ndarray] = {}  # utt_id -> features
    pbar = tqdm(total=sum(len(m[1]) for m in dialog_meta), desc="[Audio] Whisper", ncols=120)
    for (session, dialog_id), group, wav_path, video_path, time_segments in dialog_meta:
        try:
            wav, sr = torchaudio.load(wav_path)
        except Exception:
            pbar.update(len(group))
            continue
        try:
            a_feats = extractor.extract_audio_batch(wav, sr, time_segments, batch_size=16)
            for i, u in enumerate(group):
                audio_map[u.utt_id] = a_feats[i]
        except Exception as e:
            print(f"\n[Skip Audio] {session} {dialog_id}: {e}")
        pbar.update(len(group))
    pbar.close()
    extractor._unload_whisper()  # Free ~3GB VRAM

    # ── Pass 2: Video (EVA-ViT only in VRAM) ──
    print(f"\n{'='*60}")
    print(f"Pass 2/2: Extracting video features (EVA-ViT-G)")
    print(f"{'='*60}")
    video_map: Dict[str, np.ndarray] = {}  # utt_id -> features
    pbar = tqdm(total=sum(len(m[1]) for m in dialog_meta), desc="[Video] EVA-ViT", ncols=120)
    for (session, dialog_id), group, wav_path, video_path, time_segments in dialog_meta:
        try:
            v_feats = extractor.extract_video_batch(video_path, time_segments, batch_size=8)
            for i, u in enumerate(group):
                video_map[u.utt_id] = v_feats[i]
        except Exception as e:
            print(f"\n[Skip Video] {session} {dialog_id}: {e}")
        pbar.update(len(group))
    pbar.close()
    extractor._unload_eva()  # Free ~4.5GB VRAM

    # ── Assemble splits (only keep utterances that have BOTH modalities) ──
    for (session, dialog_id), group, wav_path, video_path, time_segments in dialog_meta:
        for u in group:
            if u.utt_id not in audio_map or u.utt_id not in video_map:
                continue
            if u.session == "Session5":
                split_idx, row_list = 2, test_rows
            else:
                is_valid = stable_valid_split(u.utt_id, args.valid_ratio, args.seed)
                split_idx = 1 if is_valid else 0
                row_list = valid_rows if is_valid else train_rows

            audio_splits[split_idx][u.utt_id] = audio_map[u.utt_id]
            video_splits[split_idx][u.utt_id] = video_map[u.utt_id]
            row_list.append({
                "text": u.text, "context": context_map.get(u.utt_id, "[]"),
                "speaker": u.speaker, "label": u.label, "vid_cid": u.utt_id,
            })

    del audio_map, video_map  # free memory

    # ── Post-extraction compression (if mode=="compressed") ──
    if args.mode == "compressed":
        audio_splits, video_splits = compress_features(
            audio_splits, video_splits,
            method=args.compress_method,
            out_dim=64,
            audio_target_t=157,  # Legacy config compat (ideally 64, see analysis)
            video_target_t=32,
        )

    out_pkl = os.path.join(output_dir, "iemocap_data_0610.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({"audio": audio_splits, "video": video_splits}, f, protocol=4)

    write_split_csv(os.path.join(text_dir, "iemocap_data_train.csv"), train_rows)
    write_split_csv(os.path.join(text_dir, "iemocap_data_valid.csv"), valid_rows)
    write_split_csv(os.path.join(text_dir, "iemocap_data_test.csv"), test_rows)

    stats = {
        "mode": args.mode,
        "compress_method": args.compress_method if args.mode == "compressed" else "n/a",
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

if __name__ == "__main__":
    main()
    # # Raw 模式（默认，与之前完全一致）
    # python preprocess_iemocap_emotionllama_v2.py --mode raw

    # # Compressed + PCA（推荐）
    # pip install scikit-learn
    # python preprocess_iemocap_emotionllama_v2.py --mode compressed --compress_method pca

    # # Compressed + 注意力头感知
    # python preprocess_iemocap_emotionllama_v2.py --mode compressed --compress_method head_aware

    # # Compressed + 旧方法（不推荐）
    # python preprocess_iemocap_emotionllama_v2.py --mode compressed --compress_method block_mean

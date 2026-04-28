import argparse, csv, hashlib, json, os, pickle, re, warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import av
except Exception:
    av = None
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WhisperModel

warnings.filterwarnings("ignore", category=UserWarning)

EMO_MAP_7 = {"anger":"anger","disgust":"disgust","fear":"fear","joy":"joy",
             "neutral":"neutral","surprise":"surprise","sadness":"sadness"}
EMO_MAP_3 = {"anger":"negative","disgust":"negative","fear":"negative","sadness":"negative",
             "neutral":"neutral","joy":"positive","surprise":"positive"}

@dataclass
class Utterance:
    uid: str; split: str; dialogue_id: str; utterance_id: int
    speaker: str; emotion: str; sentiment: str; text: str
    season: Optional[int] = None; episode: Optional[int] = None

# ═══════════════════════════════════════════════════════
# MELD Dataset Parsing
# ═══════════════════════════════════════════════════════
def resolve_meld_root(path: str) -> str:
    path = os.path.abspath(path)
    for cand in [path, os.path.join(path, "MELD_full_release")]:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "train_sent_emo.csv")):
            return cand
    raise FileNotFoundError(f"Cannot locate MELD root from: {path}")

def parse_csv(csv_path: str, split: str) -> List[Utterance]:
    items = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            emo = str(row.get("Emotion","")).strip().lower()
            if not emo: continue
            did = str(row.get("Dialogue_ID","")).strip()
            uid_num = int(float(row.get("Utterance_ID", 0)))
            items.append(Utterance(
                uid=f"{split}_dia{did}_utt{uid_num}", split=split, dialogue_id=did,
                utterance_id=uid_num, speaker=str(row.get("Speaker","")).strip(),
                emotion=emo, sentiment=str(row.get("Sentiment","")).strip().lower(),
                text=str(row.get("Utterance","")).strip(),
                season=int(row["Season"]) if row.get("Season") not in [None,"","nan"] else None,
                episode=int(row["Episode"]) if row.get("Episode") not in [None,"","nan"] else None,
            ))
    items.sort(key=lambda x: (int(x.dialogue_id) if x.dialogue_id.isdigit() else x.dialogue_id, x.utterance_id))
    return items

def build_context(dialog_utts: List[Utterance]) -> Dict[str, str]:
    speakers = []
    for u in dialog_utts:
        if u.speaker not in speakers: speakers.append(u.speaker)
    sp_map = {sp: chr(ord('a')+i) for i, sp in enumerate(speakers)}
    seq = [f"<{sp_map.get(u.speaker,'a')}>{u.text}" for u in dialog_utts]
    return {u.uid: json.dumps(seq, ensure_ascii=False) for u in dialog_utts}

def find_meld_video_path(meld_root: str, split: str, dialogue_id: str, utterance_id: int) -> Optional[str]:
    fname = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    split_dir = os.path.join(meld_root, split)
    if split == "valid":
        sub_dirs = [os.path.join(split_dir,"dev_splits_complete"), os.path.join(meld_root,"dev","dev_splits_complete"), split_dir]
    elif split == "train":
        sub_dirs = [os.path.join(split_dir,"train_splits"), os.path.join(split_dir,"train_splits_complete"), split_dir]
    else:
        sub_dirs = [os.path.join(split_dir,"output_repeated_splits_test"), os.path.join(split_dir,"test_splits_complete"), split_dir]
    for d in sub_dirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p): return p
    for root, _, files in os.walk(split_dir):
        if fname in files: return os.path.join(root, fname)
    return None

def read_audio_from_video(video_path: str) -> Tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(video_path)
    except Exception:
        if av is None: raise
        container = av.open(video_path)
        samples, sr = [], None
        for frame in container.decode(audio=0):
            samples.append(frame.to_ndarray()); sr = frame.sample_rate
        if not samples: return torch.zeros(1, 16000), 16000
        return torch.from_numpy(np.concatenate(samples, axis=1).astype(np.float32)), sr or 16000

# ═══════════════════════════════════════════════════════
# Feature Extraction (two-pass, memory-efficient)
# ═══════════════════════════════════════════════════════
def adaptive_downsample_with_padding(features: torch.Tensor, target_len: int) -> torch.Tensor:
    if features.shape[1] < target_len:
        features = F.pad(features, (0, 0, 0, target_len - features.shape[1]))
    features = features.transpose(1, 2)
    features = F.adaptive_avg_pool1d(features, target_len)
    return features.transpose(1, 2)

def spatiotemporal_downsample(x: torch.Tensor, target_h: int, target_w: int, target_t: int) -> torch.Tensor:
    x = x.permute(0, 4, 1, 2, 3)
    x = F.adaptive_avg_pool3d(x, (x.shape[2], target_h, target_w))
    if x.shape[2] != target_t: x = F.adaptive_avg_pool3d(x, (target_t, target_h, target_w))
    x = x.permute(0, 2, 3, 4, 1)
    b, t, h, w, c = x.shape
    return x.reshape(b, t*h*w, c)

def sample_frame_indices(clip_len: int, frame_sample_rate: int, seg_len: int) -> np.ndarray:
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len <= 0: return np.zeros((clip_len,), dtype=np.int64)
    if seg_len <= converted_len:
        return np.clip(np.linspace(0, seg_len-1, num=clip_len), 0, max(seg_len-1,0)).astype(np.int64)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    return np.clip(np.linspace(start_idx, end_idx, num=clip_len), start_idx, end_idx-1).astype(np.int64)

def read_video_cv2_abs(file_path: str, indices_abs: np.ndarray, clip_len: int = 16) -> List[np.ndarray]:
    cap = cv2.VideoCapture(file_path)
    frames = []
    if cap.isOpened():
        for idx in indices_abs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    if not frames: frames.append(np.zeros((224,224,3), dtype=np.uint8))
    while len(frames) < clip_len: frames.append(frames[-1].copy())
    return frames

class EmotionLLaMAStyleExtractor:
    """Two-pass extractor: loads only one model at a time to save VRAM."""
    def __init__(self, whisper_model: str, eva_ckpt: Optional[str], device: str):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.whisper_model_name = whisper_model
        self.eva_ckpt = eva_ckpt
        self._whisper = self._whisper_fe = self._eva = self._eva_tf = None

    def _load_whisper(self):
        if self._whisper is None:
            print("[Model] Loading Whisper...")
            self._whisper = WhisperModel.from_pretrained(self.whisper_model_name).to(self.device)
            self._whisper.eval()
            self._whisper_fe = AutoFeatureExtractor.from_pretrained(self.whisper_model_name)

    def _unload_whisper(self):
        if self._whisper is not None:
            del self._whisper; self._whisper = None
            torch.cuda.empty_cache(); print("[Model] Whisper unloaded.")

    def _load_eva(self):
        if self._eva is not None: return
        print("[Model] Loading EVA-ViT-G...")
        self._eva = create_model("eva_giant_patch14_224", pretrained=False, num_classes=0, global_pool="")
        ckpt = self.eva_ckpt or os.path.expanduser("~/.cache/torch/hub/checkpoints/eva_vit_g.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state: state = state["model"]
        self._eva.load_state_dict(state, strict=False)
        self._eva = self._eva.to(self.device); self._eva.eval()
        self._eva_tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _unload_eva(self):
        if self._eva is not None:
            del self._eva; self._eva = None
            torch.cuda.empty_cache(); print("[Model] EVA-ViT unloaded.")

    def extract_audio(self, wave: torch.Tensor, sr: int, batch_size: int = 16) -> np.ndarray:
        """Extract audio features for a single utterance. Returns (64, 1280)."""
        self._load_whisper()
        if wave.ndim == 2: wave = wave.mean(dim=0)
        if sr != 16000: wave = torchaudio.functional.resample(wave, sr, 16000)
        wave_np = wave.detach().cpu().numpy()
        inputs = self._whisper_fe([wave_np], sampling_rate=16000, return_tensors="pt", padding="max_length")
        input_features = inputs.input_features.to(device=self.device, dtype=next(self._whisper.parameters()).dtype)
        decoder_input_ids = torch.tensor([[self._whisper.config.decoder_start_token_id]], device=self.device)
        with torch.no_grad():
            out = self._whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
        feat = out.encoder_last_hidden_state  # (1, T, 1280)
        feat = adaptive_downsample_with_padding(feat, target_len=64).squeeze(0)
        return feat.float().cpu().numpy().astype(np.float32)

    def extract_video(self, video_path: str) -> np.ndarray:
        """Extract video features for a single utterance mp4. Returns (64, 1408)."""
        self._load_eva()
        import decord; decord.bridge.set_bridge('torch')
        use_decord = True
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps(); total_frames = len(vr)
        except Exception:
            use_decord = False
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        seg_len = max(1, total_frames)
        rel_idx = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=seg_len)
        if use_decord:
            try: frames = vr.get_batch(rel_idx).numpy()
            except Exception: frames = read_video_cv2_abs(video_path, rel_idx)
        else:
            frames = read_video_cv2_abs(video_path, rel_idx)
        images_tensor = torch.stack([self._eva_tf(Image.fromarray(f)) for f in frames], dim=0)
        batch_tensor = images_tensor.unsqueeze(0).view(16, 3, 224, 224).to(self.device)
        with torch.no_grad():
            feat = self._eva.forward_features(batch_tensor)
        feat = feat.view(1, 16, 257, 1408)
        feat = feat[:, :, 1:, :].reshape(1, 16, 16, 16, -1)
        feat = spatiotemporal_downsample(feat, 2, 2, 16)
        return feat[0].float().cpu().numpy().astype(np.float32)

# ═══════════════════════════════════════════════════════
# Post-extraction Compression
# ═══════════════════════════════════════════════════════
WHISPER_NUM_HEADS = 20
EVA_VIT_NUM_HEADS = 16

def _resample_time_np(x: np.ndarray, target_t: int) -> np.ndarray:
    xt = torch.from_numpy(x).float().transpose(0,1).unsqueeze(0)
    xt = F.interpolate(xt, size=target_t, mode="linear", align_corners=False)
    return xt.squeeze(0).transpose(0,1).contiguous().numpy().astype(np.float32)

def _compress_block_mean_np(x: np.ndarray, out_dim: int) -> np.ndarray:
    t, d = x.shape; blk = d // out_dim
    return x.reshape(t, out_dim, blk).mean(axis=-1).astype(np.float32)

def _compress_head_aware_np(x: np.ndarray, out_dim: int, num_heads: int) -> np.ndarray:
    t, d = x.shape; head_dim = d // num_heads
    base_out = out_dim // num_heads; remainder = out_dim % num_heads
    xt = torch.from_numpy(x).float().view(t, num_heads, head_dim)
    compressed = []
    for h in range(num_heads):
        target = base_out + (1 if h < remainder else 0)
        pooled = F.adaptive_avg_pool1d(xt[:,h,:].unsqueeze(1), target).squeeze(1)
        compressed.append(pooled)
    return torch.cat(compressed, dim=-1).numpy().astype(np.float32)

def _apply_per_sample(splits, compress_fn, target_t):
    for split_dict in splits:
        for uid in split_dict:
            split_dict[uid] = _resample_time_np(compress_fn(split_dict[uid]), target_t)

def compress_features(audio_splits, video_splits, method, out_dim=64,
                      audio_target_t=157, video_target_t=32):
    print(f"\n[Compress] method={method}, out_dim={out_dim}, audio_T={audio_target_t}, video_T={video_target_t}")
    if method == "pca":
        from sklearn.decomposition import PCA
        all_a = np.vstack([f for d in audio_splits for f in d.values()])
        pca_a = PCA(n_components=out_dim); pca_a.fit(all_a)
        print(f"[PCA Audio] {all_a.shape[1]}d -> {out_dim}d, var: {pca_a.explained_variance_ratio_.sum():.4f}")
        for sd in audio_splits:
            for uid in sd: sd[uid] = _resample_time_np(pca_a.transform(sd[uid]).astype(np.float32), audio_target_t)
        del all_a
        all_v = np.vstack([f for d in video_splits for f in d.values()])
        pca_v = PCA(n_components=out_dim); pca_v.fit(all_v)
        print(f"[PCA Video] {all_v.shape[1]}d -> {out_dim}d, var: {pca_v.explained_variance_ratio_.sum():.4f}")
        for sd in video_splits:
            for uid in sd: sd[uid] = _resample_time_np(pca_v.transform(sd[uid]).astype(np.float32), video_target_t)
        del all_v
    elif method == "head_aware":
        _apply_per_sample(audio_splits, lambda f: _compress_head_aware_np(f, out_dim, WHISPER_NUM_HEADS), audio_target_t)
        _apply_per_sample(video_splits, lambda f: _compress_head_aware_np(f, out_dim, EVA_VIT_NUM_HEADS), video_target_t)
    elif method == "block_mean":
        _apply_per_sample(audio_splits, lambda f: _compress_block_mean_np(f, out_dim), audio_target_t)
        _apply_per_sample(video_splits, lambda f: _compress_block_mean_np(f, out_dim), video_target_t)
    print("[Compress] Done.\n")

# ═══════════════════════════════════════════════════════
# CSV Output
# ═══════════════════════════════════════════════════════
def write_csv(path: str, rows: List[Dict[str, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["text","context","speaker","label","index","vid_cid"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, r in enumerate(rows):
            out = dict(r); out["index"] = i; writer.writerow(out)

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Preprocess MELD (v2, two-pass extraction)")
    parser.add_argument("--meld_root", type=str, default="/root/autodl-tmp/datasets/MELD_full_release")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/datasets/MELD")
    parser.add_argument("--split_mode", type=str, default="emotion7", choices=["emotion7","sentiment3"])
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--eva_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default="raw", choices=["raw","compressed"])
    parser.add_argument("--compress_method", type=str, default="pca", choices=["pca","head_aware","block_mean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_utts", type=int, default=None)
    parser.add_argument("--no_media", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    meld_root = resolve_meld_root(args.meld_root)
    out_dir = os.path.abspath(args.output_dir)
    text_dir = os.path.join(out_dir, "meld_text")
    os.makedirs(out_dir, exist_ok=True); os.makedirs(text_dir, exist_ok=True)
    print(f"[Info] MELD root: {meld_root}\n[Info] Output dir: {out_dir}")

    # Parse CSV
    splits = {
        "train": parse_csv(os.path.join(meld_root, "train_sent_emo.csv"), "train"),
        "valid": parse_csv(os.path.join(meld_root, "dev_sent_emo.csv"), "valid"),
        "test":  parse_csv(os.path.join(meld_root, "test_sent_emo.csv"), "test"),
    }
    if args.max_utts: splits = {k: v[:args.max_utts] for k, v in splits.items()}

    # Build context per dialogue
    context_all: Dict[str, str] = {}
    for split_name, items in splits.items():
        groups: Dict[str, List[Utterance]] = defaultdict(list)
        for u in items: groups[u.dialogue_id].append(u)
        for gid in groups: groups[gid].sort(key=lambda x: x.utterance_id)
        for gid, gutts in groups.items(): context_all.update(build_context(gutts))

    # Build utterance list with video paths
    emo_map = EMO_MAP_7 if args.split_mode == "emotion7" else EMO_MAP_3
    split_idx_map = {"train": 0, "valid": 1, "test": 2}
    utt_meta = []  # (uid, split_idx, label, text, context, speaker, video_path)
    for split_name, items in splits.items():
        si = split_idx_map[split_name]
        for u in items:
            label = emo_map.get(u.emotion)
            if label is None: continue
            vid_cid = f"dia{u.dialogue_id}_utt{u.utterance_id}"
            vpath = None if args.no_media else find_meld_video_path(meld_root, split_name, u.dialogue_id, u.utterance_id)
            utt_meta.append((vid_cid, si, label, u.text, context_all.get(u.uid,"[]"), u.speaker, vpath))

    print(f"[Info] Total utterances: {len(utt_meta)}")
    audio_splits = [dict(), dict(), dict()]
    video_splits = [dict(), dict(), dict()]
    all_rows = {0: [], 1: [], 2: []}

    if args.no_media:
        # Placeholder features only
        for vid_cid, si, label, text, ctx, speaker, _ in utt_meta:
            audio_splits[si][vid_cid] = np.zeros((64, 1280), dtype=np.float32)
            video_splits[si][vid_cid] = np.zeros((64, 1408), dtype=np.float32)
            all_rows[si].append({"text":text,"context":ctx,"speaker":speaker,"label":label,"vid_cid":vid_cid})
    else:
        extractor = EmotionLLaMAStyleExtractor(args.whisper_model, args.eva_ckpt, args.device)

        # ── Pass 1: Audio (Whisper) ──
        print(f"\n{'='*60}\nPass 1/2: Extracting audio features (Whisper)\n{'='*60}")
        audio_map: Dict[str, np.ndarray] = {}
        pbar = tqdm(total=len(utt_meta), desc="[Audio] Whisper", ncols=120)
        for vid_cid, si, label, text, ctx, speaker, vpath in utt_meta:
            if vpath:
                try:
                    wav, sr = read_audio_from_video(vpath)
                    audio_map[vid_cid] = extractor.extract_audio(wav, sr)
                except Exception as e:
                    print(f"\n[Skip Audio] {vid_cid}: {e}")
            pbar.update(1)
        pbar.close()
        extractor._unload_whisper()

        # ── Pass 2: Video (EVA-ViT) ──
        print(f"\n{'='*60}\nPass 2/2: Extracting video features (EVA-ViT-G)\n{'='*60}")
        video_map: Dict[str, np.ndarray] = {}
        pbar = tqdm(total=len(utt_meta), desc="[Video] EVA-ViT", ncols=120)
        for vid_cid, si, label, text, ctx, speaker, vpath in utt_meta:
            if vpath:
                try:
                    video_map[vid_cid] = extractor.extract_video(vpath)
                except Exception as e:
                    print(f"\n[Skip Video] {vid_cid}: {e}")
            pbar.update(1)
        pbar.close()
        extractor._unload_eva()

        # ── Assemble ──
        for vid_cid, si, label, text, ctx, speaker, vpath in utt_meta:
            a_feat = audio_map.get(vid_cid, np.zeros((64, 1280), dtype=np.float32))
            v_feat = video_map.get(vid_cid, np.zeros((64, 1408), dtype=np.float32))
            audio_splits[si][vid_cid] = a_feat
            video_splits[si][vid_cid] = v_feat
            all_rows[si].append({"text":text,"context":ctx,"speaker":speaker,"label":label,"vid_cid":vid_cid})
        del audio_map, video_map

    # ── Post-extraction compression ──
    if args.mode == "compressed":
        compress_features(audio_splits, video_splits, method=args.compress_method,
                          out_dim=64, audio_target_t=157, video_target_t=32)

    # ── Save ──
    pkl_path = os.path.join(out_dir, "meld_data_0610.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"audio": audio_splits, "video": video_splits}, f, protocol=4)

    name_map = {0: "train", 1: "valid", 2: "test"}
    for si in range(3):
        write_csv(os.path.join(text_dir, f"meld_data_{name_map[si]}.csv"), all_rows[si])

    stats = {
        "meld_root": meld_root, "output_dir": out_dir, "split_mode": args.split_mode,
        "mode": args.mode, "compress_method": args.compress_method if args.mode=="compressed" else "n/a",
        "num_train": len(all_rows[0]), "num_valid": len(all_rows[1]), "num_test": len(all_rows[2]),
        "pkl_path": pkl_path, "csv_dir": text_dir,
        "audio_shape": [64,1280] if args.mode=="raw" else [157,64],
        "video_shape": [64,1408] if args.mode=="raw" else [32,64],
        "note": "Pickle: data['audio'][split] and data['video'][split] are dicts keyed by vid_cid."
    }
    with open(os.path.join(out_dir, "preprocess_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] MELD preprocessing complete.")
    print(f"  PKL: {pkl_path}")
    print(f"  CSV: {text_dir}")
    for si in range(3): print(f"  {name_map[si]}: {len(all_rows[si])} samples")

if __name__ == "__main__":
    main()
    # Raw mode (default):
    # python preprocess_meld_emotionllama_v2.py --mode raw
    #
    # Compressed + PCA:
    # python preprocess_meld_emotionllama_v2.py --mode compressed --compress_method pca
    #
    # Compressed + head-aware:
    # python preprocess_meld_emotionllama_v2.py --mode compressed --compress_method head_aware
    #
    # No media (text-only placeholders):
    # python preprocess_meld_emotionllama_v2.py --no_media

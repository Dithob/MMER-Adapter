"""
Detailed Classification Analysis Toolkit for MMER-Adapter.

Provides:
  - Per-class Precision / Recall / F1 / Support
  - WA (Weighted Accuracy), UA (Unweighted Accuracy), WF1, Macro-F1
  - Confusion matrix heatmap (PNG)
  - t-SNE emotion clustering scatter plot (PNG)
  - t-SNE modality separation scatter plot (PNG) — paper Figure 4 style

Usage:
    from utils.analysis import detailed_classification_analysis
    result = detailed_classification_analysis(
        y_true, y_pred, features, modality_features, label_names,
        save_dir, tag, logger, timestamp)
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.manifold import TSNE

logger = logging.getLogger('MSA')

# ── Color palettes ──
# Emotion classes: high-contrast, distinguishable
EMOTION_COLORS = [
    '#E74C3C',  # Red       — angry / anger
    '#F39C12',  # Orange    — surprise / excited
    '#2ECC71',  # Green     — happy / joy
    '#3498DB',  # Blue      — sad / sadness
    '#9B59B6',  # Purple    — fear / frustrated
    '#1ABC9C',  # Teal      — neutral
    '#E67E22',  # Dark Orange — disgust
    '#34495E',  # Dark Gray  — other
]

# Modality colors (matching paper Figure 4: red=text, blue=audio, green=video)
MODALITY_COLORS = {
    'fusion': '#E74C3C',  # Red
    'audio':  '#3498DB',  # Blue
    'video':  '#2ECC71',  # Green
}
MODALITY_LABELS = {
    'fusion': 'Text (Fusion)',
    'audio':  'Audio',
    'video':  'Video',
}


def _try_chinese_font():
    """Try to find a Chinese-capable font for matplotlib labels."""
    candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'STHeiti', 'PingFang SC'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def detailed_classification_analysis(
    y_true, y_pred, features, label_names, save_dir, tag,
    modality_features=None, log=None, timestamp=None
):
    """
    Run full classification analysis and save results.

    Args:
        y_true: list/array of ground-truth label indices (int)
        y_pred: list/array of predicted label indices (int)
        features: np.ndarray (N, D) — fusion features for emotion t-SNE.
                  Can be None to skip.
        label_names: list of str — emotion class names in index order
        save_dir: str — directory to save output PNGs
        tag: str — prefix for filenames (e.g. 'hmmem-chatglm3-meld-TEST')
        modality_features: dict with keys 'audio', 'video', 'fusion',
                           each np.ndarray (N, 256). For modality separation
                           t-SNE (Figure 4 style). Can be None to skip.
        log: logger instance (optional)
        timestamp: str — timestamp suffix for filenames (optional)

    Returns:
        dict with keys: WA, UA, WF1, Macro_F1, per_class, cm_path, tsne_path
    """
    if log is None:
        log = logger

    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    ts_suffix = f'-{timestamp}' if timestamp else ''

    # ── 1. Summary metrics ──
    wa = accuracy_score(y_true, y_pred)
    ua = recall_score(y_true, y_pred, average='macro', zero_division=0)
    wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # ── 2. Per-class metrics ──
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))), zero_division=0
    )

    per_class = []
    for i, name in enumerate(label_names):
        per_class.append({
            'label': name,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i]),
        })

    # ── 3. Log the report ──
    log.info("=" * 60)
    log.info("         Detailed Classification Analysis")
    log.info("=" * 60)
    log.info(f"{'Class':>14s}  {'Prec':>7s}  {'Recall':>7s}  {'F1':>7s}  {'Support':>8s}")
    log.info("-" * 60)
    for pc in per_class:
        log.info(f"{pc['label']:>14s}  {pc['precision']:7.4f}  {pc['recall']:7.4f}  "
                 f"{pc['f1']:7.4f}  {pc['support']:8d}")
    log.info("-" * 60)
    log.info(f"  WA  (Weighted Accuracy) : {wa:.4f}")
    log.info(f"  UA  (Unweighted Accuracy): {ua:.4f}")
    log.info(f"  WF1 (Weighted F1)       : {wf1:.4f}")
    log.info(f"  Macro-F1                : {macro_f1:.4f}")
    log.info("=" * 60)

    # ── 4. Confusion matrix heatmap ──
    cm_path = os.path.join(save_dir, f'{tag}-confusion_matrix{ts_suffix}.png')
    _plot_confusion_matrix(y_true, y_pred, label_names, cm_path, tag)
    log.info(f"Confusion matrix saved → {cm_path}")

    # ── 5. Emotion t-SNE clustering ──
    tsne_path = None
    if features is not None and len(features) > 0:
        tsne_path = os.path.join(save_dir, f'{tag}-tsne_emotion{ts_suffix}.png')
        _plot_emotion_tsne(features, y_true, label_names, tsne_path, tag)
        log.info(f"Emotion t-SNE saved → {tsne_path}")

    # ── 6. Modality separation t-SNE (Figure 4 style) ──
    modality_tsne_path = None
    if modality_features is not None:
        modality_tsne_path = os.path.join(save_dir, f'{tag}-tsne_modality{ts_suffix}.png')
        _plot_modality_tsne(modality_features, modality_tsne_path, tag)
        log.info(f"Modality t-SNE saved → {modality_tsne_path}")

    return {
        'WA': wa,
        'UA': ua,
        'WF1': wf1,
        'Macro_F1': macro_f1,
        'per_class': per_class,
        'cm_path': cm_path,
        'tsne_path': tsne_path,
    }


def _plot_confusion_matrix(y_true, y_pred, label_names, save_path, tag):
    """
    Plot confusion matrix heatmap with both percentages and raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.2),
                                     max(5, len(label_names) * 1.0)))

    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    cax = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.1%}\n({cm[i, j]})",
                    ha="center", va="center", color=text_color,
                    fontsize=max(7, 12 - len(label_names)))

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right', **font_props)
    ax.set_yticklabels(label_names, **font_props)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — {tag}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Emotion Clustering t-SNE (colored by emotion label)
# ═══════════════════════════════════════════════════════════

def _plot_emotion_tsne(features, labels, label_names, save_path, tag,
                       perplexity=30, max_samples=5000):
    """
    Clean t-SNE scatter plot colored by emotion class.
    Publication-quality: no grid, no contours, minimal axes.
    """
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=int)

    # Subsample for performance
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Handle NaN/Inf
    valid_mask = np.all(np.isfinite(features), axis=1)
    if not np.all(valid_mask):
        features = features[valid_mask]
        labels = labels[valid_mask]

    if len(features) < 10:
        logger.warning("Too few valid samples for t-SNE, skipping.")
        return

    effective_perplexity = min(perplexity, max(5, len(features) // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto',
        max_iter=1500,
    )
    embeddings = tsne.fit_transform(features)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    chinese_font = _try_chinese_font()
    legend_prop = {}
    if chinese_font:
        legend_prop = {'family': chinese_font}

    colors = EMOTION_COLORS[:len(label_names)]
    unique_labels = sorted(set(labels))

    # Adaptive point size: larger for fewer points
    n_total = len(features)
    point_size = max(6, min(50, 3000 / max(n_total, 1)))

    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        n_pts = mask.sum()
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[idx % len(colors)],
            label=f'{label_names[idx]} ({n_pts})',
            alpha=0.7,
            s=point_size,
            edgecolors='none',
            rasterized=True,  # faster rendering for many points
        )

    ax.legend(
        loc='best', fontsize=9, framealpha=0.9,
        markerscale=max(1, 8 / point_size),
        prop=legend_prop if legend_prop else None,
        title='Emotion', title_fontsize=10,
    )
    ax.set_title(f't-SNE Emotion Clusters — {tag}',
                 fontsize=13, fontweight='bold')

    # Clean minimal style — no ticks, no grid
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Modality Separation t-SNE (paper Figure 4 style)
#  Red = Fusion/Text, Blue = Audio, Green = Video
# ═══════════════════════════════════════════════════════════

def _plot_modality_tsne(modality_features, save_path, tag,
                        perplexity=30, max_samples_per_modality=3000):
    """
    t-SNE visualization showing separation of text/audio/video modality
    features in learned representation space.

    Mimics ATGFB-MFF Figure 4: dense clusters, 3 colors (red/blue/green),
    clean white background, no grid, no axes labels.

    Args:
        modality_features: dict with keys 'fusion', 'audio', 'video',
                           each np.ndarray (N, D).
        save_path: str
        tag: str
        perplexity: int
        max_samples_per_modality: int — cap per modality for performance
    """
    # Collect and label each modality
    all_feats = []
    all_labels = []  # 0=fusion, 1=audio, 2=video
    modality_order = ['fusion', 'audio', 'video']
    modality_counts = {}

    for mod_idx, mod_name in enumerate(modality_order):
        feats = modality_features.get(mod_name)
        if feats is None or len(feats) == 0:
            continue
        feats = np.array(feats, dtype=np.float32)

        # Handle NaN/Inf
        valid = np.all(np.isfinite(feats), axis=1)
        feats = feats[valid]

        # Subsample
        if len(feats) > max_samples_per_modality:
            idx = np.random.choice(len(feats), max_samples_per_modality, replace=False)
            feats = feats[idx]

        modality_counts[mod_name] = len(feats)
        all_feats.append(feats)
        all_labels.append(np.full(len(feats), mod_idx, dtype=int))

    if len(all_feats) < 2:
        logger.warning("Need at least 2 modalities for modality t-SNE, skipping.")
        return

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if len(all_feats) < 20:
        logger.warning("Too few samples for modality t-SNE, skipping.")
        return

    effective_perplexity = min(perplexity, max(5, len(all_feats) // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto',
        max_iter=1500,
    )
    embeddings = tsne.fit_transform(all_feats)

    # ── Publication-quality plot (Figure 4 style) ──
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    n_total = len(all_feats)
    point_size = max(4, min(30, 5000 / max(n_total, 1)))

    for mod_idx, mod_name in enumerate(modality_order):
        if mod_name not in modality_counts:
            continue
        mask = all_labels == mod_idx
        color = MODALITY_COLORS[mod_name]
        label = MODALITY_LABELS[mod_name]
        n_pts = mask.sum()

        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=color,
            label=f'{label} ({n_pts})',
            alpha=0.6,
            s=point_size,
            edgecolors='none',
            rasterized=True,
        )

    ax.legend(
        loc='best', fontsize=11, framealpha=0.9,
        markerscale=max(1, 10 / point_size),
        title='Modality', title_fontsize=12,
    )

    # Dataset name extraction for clean title
    dataset_short = tag.split('-')[-2] if '-' in tag else tag
    ax.set_title(f't-SNE Modality Separation — {dataset_short.upper()}',
                 fontsize=14, fontweight='bold')

    # Clean style: no ticks, no grid, no spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

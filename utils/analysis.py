"""
Detailed Classification Analysis Toolkit for MMER-Adapter.

Provides:
  - Per-class Precision / Recall / F1 / Support
  - WA (Weighted Accuracy), UA (Unweighted Accuracy), WF1, Macro-F1
  - Confusion matrix heatmap (PNG)
  - t-SNE feature clustering scatter plot (PNG)

Usage:
    from utils.analysis import detailed_classification_analysis
    detailed_classification_analysis(y_true, y_pred, features, label_names, save_dir, tag, logger)
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
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

logger = logging.getLogger('MSA')

# ── Aesthetic color palette for emotion categories ──
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
    y_true, y_pred, features, label_names, save_dir, tag, log=None
):
    """
    Run full classification analysis and save results.

    Args:
        y_true: list/array of ground-truth label indices (int)
        y_pred: list/array of predicted label indices (int)
        features: np.ndarray of shape (N, D) — fusion features for t-SNE
                  Can be None to skip clustering visualization.
        label_names: list of str — emotion class names in index order
        save_dir: str — directory to save output PNGs
        tag: str — prefix for output filenames (e.g. 'hmmem-meld')
        log: logger instance (optional, defaults to module logger)

    Returns:
        dict with keys: WA, UA, WF1, Macro_F1, per_class (list of dicts)
    """
    if log is None:
        log = logger

    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

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
    cm_path = os.path.join(save_dir, f'{tag}-confusion_matrix.png')
    _plot_confusion_matrix(y_true, y_pred, label_names, cm_path, tag)
    log.info(f"Confusion matrix saved → {cm_path}")

    # ── 5. t-SNE clustering ──
    if features is not None and len(features) > 0:
        tsne_path = os.path.join(save_dir, f'{tag}-tsne_clusters.png')
        _plot_tsne_clusters(features, y_true, label_names, tsne_path, tag)
        log.info(f"t-SNE cluster plot saved → {tsne_path}")

    return {
        'WA': wa,
        'UA': ua,
        'WF1': wf1,
        'Macro_F1': macro_f1,
        'per_class': per_class,
    }


def _plot_confusion_matrix(y_true, y_pred, label_names, save_path, tag):
    """
    Plot confusion matrix heatmap with both percentages and raw counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.2),
                                     max(5, len(label_names) * 1.0)))

    # Check if labels need Chinese font
    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    # Draw heatmap manually (avoid seaborn dependency)
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells with percentage + count
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


def _plot_tsne_clusters(features, labels, label_names, save_path, tag,
                        perplexity=30, max_samples=5000):
    """
    t-SNE 2D visualization of fusion features colored by emotion label.

    Args:
        features: np.ndarray (N, D) — feature vectors
        labels: np.ndarray (N,) — integer labels
        label_names: list of str
        save_path: str
        tag: str
        perplexity: int — t-SNE perplexity
        max_samples: int — subsample if dataset is too large
    """
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=int)

    # Subsample for performance
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Adjust perplexity if sample count is too small
    effective_perplexity = min(perplexity, max(5, len(features) // 4))

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                random_state=42, init='pca', learning_rate='auto')
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Check if labels need Chinese font
    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    colors = EMOTION_COLORS[:len(label_names)]
    unique_labels = sorted(set(labels))

    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[idx % len(colors)],
            label=label_names[idx],
            alpha=0.55, s=18, edgecolors='none'
        )
        # Draw cluster centroid
        centroid = embeddings[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1],
                   c=colors[idx % len(colors)],
                   marker='X', s=200, edgecolors='black', linewidths=1.2,
                   zorder=10)

    ax.legend(loc='best', fontsize=10, framealpha=0.8, prop=font_props if font_props else None)
    ax.set_title(f't-SNE Emotion Feature Clusters — {tag}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax.grid(True, alpha=0.2)

    # Remove axis values (t-SNE dimensions are not meaningful)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

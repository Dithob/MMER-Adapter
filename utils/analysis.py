"""
Detailed Classification Analysis Toolkit for MMER-Adapter.

Provides:
  - Per-class Precision / Recall / F1 / Support
  - WA (Weighted Accuracy), UA (Unweighted Accuracy), WF1, Macro-F1
  - Confusion matrix heatmap (PNG + SVG)
  - t-SNE emotion clustering scatter plot (PNG + SVG) — true-label & predicted-label
  - t-SNE modality separation scatter plot (PNG + SVG) — paper Figure 4 style

Usage:
    from utils.analysis import detailed_classification_analysis
    result = detailed_classification_analysis(
        y_true, y_pred, features, modality_features, label_names,
        save_dir, tag, logger, timestamp)
"""

import os
import logging
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Global font: Times New Roman for publication quality ──
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # math font consistent with TNR
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_fscore_support, confusion_matrix
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

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


def _save_multi_format(fig, base_path, dpi=300, facecolor='white'):
    """Save figure in PNG + SVG formats for publication quality.

    Args:
        fig: matplotlib Figure
        base_path: str — path ending with .png (used as the base)
        dpi: int — DPI for raster format (PNG)
        facecolor: str — background color
    """
    for ext in ['.png', '.svg']:
        path = base_path.replace('.png', ext)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=facecolor,
                    format=ext.lstrip('.'))


def detailed_classification_analysis(
    y_true, y_pred, features, label_names, save_dir, tag,
    modality_features=None, log=None, timestamp=None,
    max_per_class=150,
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
        max_per_class: int — controls balanced sampling for emotion t-SNE.
            > 0: each class capped at this number (balanced, prevents neutral domination)
              0: no sampling, use ALL samples (denser plot)
            CLI override: --tsne_max_per_class 0

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
    log.info(f"Confusion matrix saved → {cm_path} (+.svg)")

    # ── 4b. Clean confusion matrix (percentage-only, for paper figures) ──
    cm_clean_path = os.path.join(save_dir, f'{tag}-confusion_matrix_clean{ts_suffix}.png')
    _plot_confusion_matrix_clean(y_true, y_pred, label_names, cm_clean_path, tag)
    log.info(f"Clean confusion matrix saved → {cm_clean_path} (+.svg)")

    # ── 5. Emotion t-SNE clustering ──
    tsne_path = None
    emotion_cache = None  # will hold (embeddings, true_labels_bal, pred_labels_bal)
    if features is not None and len(features) > 0:
        tsne_path = os.path.join(save_dir, f'{tag}-tsne_emotion_true{ts_suffix}.png')

        emotion_cache = _plot_emotion_tsne(
            features, true_labels=y_true, pred_labels=y_pred,
            label_names=label_names, save_path=tsne_path, tag=tag,
            max_per_class=max_per_class,
        )
        log.info(f"Emotion t-SNE saved → {tsne_path} (+.svg, +pred)")

    # ── 6. Modality separation t-SNE (Figure 4 style) ──
    modality_tsne_path = None
    modality_cache = None  # will hold (embeddings, labels)
    if modality_features is not None:
        modality_tsne_path = os.path.join(save_dir, f'{tag}-tsne_modality{ts_suffix}.png')
        modality_cache = _plot_modality_tsne(modality_features, modality_tsne_path, tag)
        log.info(f"Modality t-SNE saved → {modality_tsne_path} (+.svg)")

    # ── 7. Save npz with features + cached t-SNE embeddings for fast replotting ──
    if features is not None and len(features) > 0:
        npz_path = os.path.join(save_dir, f'{tag}-tsne_features{ts_suffix}.npz')
        save_dict = dict(
            features=features,
            labels=y_true,
            pred_labels=y_pred,
        )
        # Cached emotion t-SNE embeddings (skip recomputation on replot)
        if emotion_cache is not None:
            emb, tl, pl = emotion_cache
            save_dict['emo_embeddings'] = emb
            save_dict['emo_true_labels'] = tl
            save_dict['emo_pred_labels'] = pl
        # Modality features + cached modality t-SNE embeddings
        if modality_features is not None:
            for mod_name in ('audio', 'video', 'fusion'):
                mod_feat = modality_features.get(mod_name)
                if mod_feat is not None:
                    save_dict[f'mod_{mod_name}'] = np.array(mod_feat)
        if modality_cache is not None:
            mod_emb, mod_lbl = modality_cache
            save_dict['mod_embeddings'] = mod_emb
            save_dict['mod_labels'] = mod_lbl
        np.savez(npz_path, **save_dict)
        log.info(f"t-SNE features + cached embeddings saved → {npz_path}")

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
    Fixed canvas size (8x7) ensures MELD-7 and IEMOCAP-4/6 produce
    identically-dimensioned figures with consistent font sizes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    n = len(label_names)
    # Fixed figure size — guarantees identical canvas across datasets (4/6/7 classes)
    fig, ax = plt.subplots(figsize=(8, 7))

    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    cax = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_norm.max() / 2.0
    # Fixed cell font size — independent of class count for cross-dataset consistency
    cell_fontsize = 12
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.1%}\n({cm[i, j]})",
                    ha="center", va="center", color=text_color,
                    fontsize=cell_fontsize, fontweight='medium')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=12, **font_props)
    ax.set_yticklabels(label_names, fontsize=12, **font_props)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    # Extract dataset name from tag (e.g. 'hmmem-chatglm3-iemocap6-TEST' → 'IEMOCAP')
    parts = tag.split('-')
    dataset_name = re.sub(r'\d+$', '', parts[2]).upper() if len(parts) >= 3 else tag
    ax.set_title(f'Normalized Confusion Matrix of {dataset_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    _save_multi_format(fig, save_path, dpi=300)
    plt.close(fig)


def _plot_confusion_matrix_clean(y_true, y_pred, label_names, save_path, tag):
    """
    Clean confusion matrix: cells show only the percentage number (no sample
    count, no '%' symbol). The '%' is shown on the colorbar instead.
    Designed for paper figures where space inside cells is limited.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    n = len(label_names)
    fig, ax = plt.subplots(figsize=(8, 7))

    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    # Display as 0-100 scale so cell text is just a number like "82.8"
    cax = ax.imshow(cm_norm * 100, interpolation='nearest', cmap='Blues',
                    vmin=0, vmax=100)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy (%)', fontsize=12)

    thresh = cm_norm.max() / 2.0
    # Larger font since cells only contain one short number
    cell_fontsize = 14
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            text_color = 'white' if val > thresh else 'black'
            ax.text(j, i, f'{val * 100:.1f}',
                    ha='center', va='center', color=text_color,
                    fontsize=cell_fontsize, fontweight='medium')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=12, **font_props)
    ax.set_yticklabels(label_names, fontsize=12, **font_props)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    # Extract dataset name (strip trailing digits: iemocap6 → IEMOCAP)
    parts = tag.split('-')
    dataset_name = re.sub(r'\d+$', '', parts[2]).upper() if len(parts) >= 3 else tag
    ax.set_title(f'Normalized Confusion Matrix of {dataset_name}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    _save_multi_format(fig, save_path, dpi=300)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
#  Emotion Clustering t-SNE (colored by emotion label)
#  Optimized: balanced sampling, preprocessing, cosine metric
# ═══════════════════════════════════════════════════════════

def _balanced_subsample(features, labels, max_per_class=150, rng=None):
    """Subsample each class to at most max_per_class for balanced t-SNE.

    This prevents high-frequency classes (e.g. neutral) from dominating
    the t-SNE layout and drowning out minority classes.

    Args:
        features: np.ndarray (N, D)
        labels: np.ndarray (N,)
        max_per_class: int — max samples per class
        rng: np.random.Generator (optional)

    Returns:
        subsampled features, subsampled labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, max_per_class, replace=False)
        indices.append(cls_idx)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return features[indices], labels[indices], indices


def _preprocess_features(features):
    """Preprocess features for t-SNE: StandardScaler → L2 normalize → PCA.

    This pipeline ensures:
    1. StandardScaler: zero mean, unit variance per dimension
    2. L2 normalize: project onto unit hypersphere (cosine-friendly)
    3. PCA: reduce dimensionality to remove noise dimensions

    Args:
        features: np.ndarray (N, D)

    Returns:
        processed features: np.ndarray (N, pca_dim)
    """
    # Step 1: Standard scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Step 2: L2 normalization (row-wise)
    features = normalize(features, norm='l2', axis=1)

    # Step 3: PCA dimensionality reduction
    pca_dim = min(50, features.shape[1], features.shape[0] - 1)
    if pca_dim < features.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=42)
        features = pca.fit_transform(features)

    return features


def _plot_emotion_tsne(features, true_labels, pred_labels, label_names,
                       save_path, tag, max_per_class=150):
    """
    Publication-quality t-SNE scatter plot colored by emotion class.

    Generates two plots:
      1. True-label coloring (primary figure)
      2. Predicted-label coloring (secondary figure for comparison)

    Args:
      max_per_class: int — controls balanced sampling.
          > 0: each class capped at this number (balanced)
            0: no sampling, use ALL samples (denser plot)

    Pipeline:
      [Optional balanced subsampling] → StandardScaler → L2 → PCA → t-SNE (cosine)
    """
    features = np.array(features, dtype=np.float32)
    true_labels = np.array(true_labels, dtype=int)
    pred_labels = np.array(pred_labels, dtype=int)

    # Handle NaN/Inf
    valid_mask = np.all(np.isfinite(features), axis=1)
    if not np.all(valid_mask):
        features = features[valid_mask]
        true_labels = true_labels[valid_mask]
        pred_labels = pred_labels[valid_mask]

    if len(features) < 10:
        logger.warning("Too few valid samples for t-SNE, skipping.")
        return None

    # ── 1. Balanced subsampling (skip when max_per_class <= 0) ──
    if max_per_class > 0:
        features_bal, true_labels_bal, bal_idx = _balanced_subsample(
            features, true_labels, max_per_class=max_per_class
        )
        pred_labels_bal = pred_labels[bal_idx]
        logger.info(f"t-SNE balanced sampling: {len(features)} → {len(features_bal)} "
                    f"(max_per_class={max_per_class})")
    else:
        features_bal = features
        true_labels_bal = true_labels
        pred_labels_bal = pred_labels
        logger.info(f"t-SNE using ALL {len(features)} samples (no balanced sampling)")

    if len(features_bal) < 10:
        logger.warning("Too few samples for t-SNE, skipping.")
        return None

    # ── 2. Feature preprocessing: StandardScaler → L2 → PCA ──
    features_proc = _preprocess_features(features_bal)

    # ── 3. t-SNE with cosine metric and optimized hyperparams ──
    n_samples = len(features_proc)
    # Adaptive perplexity: heuristic based on sample count
    perplexity = int(np.clip(n_samples / 5, 5, 50))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric='cosine',
        early_exaggeration=12,
        random_state=42,
        init='random',  # 'pca' not supported with cosine metric
        learning_rate='auto',
        max_iter=3000,
    )
    embeddings = tsne.fit_transform(features_proc)

    # ── 4. Plot: True labels (primary figure) ──
    _draw_tsne_scatter(embeddings, true_labels_bal, label_names,
                       save_path, tag, title_suffix='True Labels')

    # ── 5. Plot: Predicted labels (secondary figure) ──
    pred_save_path = save_path.replace('_true', '_pred')
    _draw_tsne_scatter(embeddings, pred_labels_bal, label_names,
                       pred_save_path, tag, title_suffix='Predicted Labels')

    # ── 6. Plot: Correctly predicted samples only (supplementary) ──
    correct_mask = true_labels_bal == pred_labels_bal
    if correct_mask.sum() >= 10:
        correct_save_path = save_path.replace('_true', '_correct')
        _draw_tsne_scatter(embeddings[correct_mask], true_labels_bal[correct_mask],
                           label_names, correct_save_path, tag,
                           title_suffix='Correct Predictions Only')

    # Return cached data for npz storage
    return (embeddings, true_labels_bal, pred_labels_bal)


def _draw_tsne_scatter(embeddings, labels, label_names, save_path, tag,
                       title_suffix=''):
    """Draw a single t-SNE scatter plot and save in PNG + SVG.

    Style matches publication convention: visible axes with t-SNE coordinates,
    semi-transparent legend, no title (use figure caption in paper).

    Args:
        embeddings: np.ndarray (N, 2) — t-SNE coordinates
        labels: np.ndarray (N,) — class labels for coloring
        label_names: list of str
        save_path: str — base path (.png)
        tag: str — experiment tag
        title_suffix: str — appended to plot title
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    colors = EMOTION_COLORS[:len(label_names)]
    unique_labels = sorted(set(labels))

    # Point size: larger for fewer points, with higher baseline than before
    n_total = len(embeddings)
    point_size = max(25, min(80, 6000 / max(n_total, 1)))

    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[idx % len(colors)],
            label=label_names[idx],
            alpha=0.75,
            s=point_size,
            edgecolors='white',
            linewidths=0.3,
        )

    # Semi-transparent legend at lower-left, matching reference paper style
    ax.legend(
        loc='lower left', fontsize=12, framealpha=0.6,
        markerscale=max(1.0, 10 / point_size),
        title='Emotion Classes', title_fontsize=13,
        edgecolor='gray',
    )

    # Keep axis ticks visible (t-SNE coordinates) — matching reference style
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    _save_multi_format(fig, save_path, dpi=300, facecolor='white')
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

    _draw_modality_scatter(embeddings, all_labels, modality_order,
                           modality_counts, save_path)

    # Return cached data for npz storage
    return (embeddings, all_labels)


def _draw_modality_scatter(embeddings, all_labels, modality_order,
                           modality_counts, save_path):
    """Draw modality t-SNE scatter plot. Separated for reuse in replotting."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    n_total = len(embeddings)
    point_size = max(25, min(80, 6000 / max(n_total, 1)))

    modality_order_full = ['fusion', 'audio', 'video']
    for mod_idx, mod_name in enumerate(modality_order_full):
        if mod_name not in modality_counts:
            continue
        mask = all_labels == mod_idx
        color = MODALITY_COLORS[mod_name]
        label = MODALITY_LABELS[mod_name]

        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=color,
            label=label,
            alpha=0.75,
            s=point_size,
            edgecolors='white',
            linewidths=0.3,
        )

    # Semi-transparent legend at lower-left, matching emotion t-SNE style
    ax.legend(
        loc='lower left', fontsize=12, framealpha=0.6,
        markerscale=max(1.0, 10 / point_size),
        title='Modality', title_fontsize=13,
        edgecolor='gray',
    )

    # Keep axis ticks visible (t-SNE coordinates)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    _save_multi_format(fig, save_path, dpi=300, facecolor='white')
    plt.close(fig)

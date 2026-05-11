"""
Detailed Classification Analysis Toolkit for MMER-Adapter.

Provides:
  - Per-class Precision / Recall / F1 / Support
  - WA (Weighted Accuracy), UA (Unweighted Accuracy), WF1, Macro-F1
  - Confusion matrix heatmap (PNG)
  - t-SNE feature clustering scatter plot (PNG) — with KDE density contours
    and confidence ellipses (inspired by ATGFB-MFF Figure 4 style)

Usage:
    from utils.analysis import detailed_classification_analysis
    result = detailed_classification_analysis(
        y_true, y_pred, features, label_names, save_dir, tag, logger, timestamp)
    # result now contains 'cm_path' and 'tsne_path' keys
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    precision_recall_fscore_support, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

logger = logging.getLogger('MSA')

# ── Aesthetic color palette for emotion categories ──
# High-contrast, colorblind-friendly palette with good separation
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

# Softer fill colors (with alpha) for density contours
EMOTION_FILL_COLORS = [
    '#E74C3C30', '#F39C1230', '#2ECC7130', '#3498DB30',
    '#9B59B630', '#1ABC9C30', '#E67E2230', '#34495E30',
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
    y_true, y_pred, features, label_names, save_dir, tag, log=None, timestamp=None
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
        tag: str — prefix for output filenames (e.g. 'hmmem-chatglm3-meld-TEST')
        log: logger instance (optional, defaults to module logger)
        timestamp: str — timestamp string to add to filename (e.g. '20260428_101530').
                   If None, files will be saved without timestamp (overwrite mode).

    Returns:
        dict with keys: WA, UA, WF1, Macro_F1, per_class (list of dicts),
                        cm_path (str), tsne_path (str or None)
    """
    if log is None:
        log = logger

    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Build filename suffix
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

    # ── 5. t-SNE clustering ──
    tsne_path = None
    if features is not None and len(features) > 0:
        tsne_path = os.path.join(save_dir, f'{tag}-tsne_clusters{ts_suffix}.png')
        _plot_tsne_clusters(features, y_true, label_names, tsne_path, tag)
        log.info(f"t-SNE cluster plot saved → {tsne_path}")

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


def _confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Draw a confidence ellipse (covariance ellipse) for 2D point cloud.
    
    Args:
        x, y: 1D arrays of coordinates
        ax: matplotlib Axes
        n_std: number of standard deviations for ellipse radius
        facecolor: fill color for ellipse
        **kwargs: passed to matplotlib.patches.Ellipse
    """
    if len(x) < 3:
        return None
    
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return None
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Ensure non-negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Angle of the first eigenvector
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Width and height (2 * n_std * sqrt(eigenvalue))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height,
        angle=angle,
        facecolor=facecolor,
        **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def _plot_tsne_clusters(features, labels, label_names, save_path, tag,
                        perplexity=30, max_samples=5000):
    """
    t-SNE 2D visualization of fusion features colored by emotion label.
    
    Inspired by ATGFB-MFF (Figure 4): clean cluster separation with
    confidence ellipses, KDE density contours, and publication-quality styling.

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

    # Handle NaN/Inf in features
    valid_mask = np.all(np.isfinite(features), axis=1)
    if not np.all(valid_mask):
        features = features[valid_mask]
        labels = labels[valid_mask]

    if len(features) < 10:
        logger.warning("Too few valid samples for t-SNE visualization, skipping.")
        return

    # Adjust perplexity if sample count is too small
    effective_perplexity = min(perplexity, max(5, len(features) // 4))

    # t-SNE with tuned parameters for better cluster separation
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        init='pca',
        learning_rate='auto',
        max_iter=1000,
        metric='cosine',  # cosine distance often works better for high-dim embeddings
    )
    embeddings = tsne.fit_transform(features)

    # Normalize embeddings to [-1, 1] for cleaner plot
    emb_min = embeddings.min(axis=0)
    emb_max = embeddings.max(axis=0)
    emb_range = emb_max - emb_min
    emb_range[emb_range == 0] = 1
    embeddings = 2 * (embeddings - emb_min) / emb_range - 1

    # ── Plot setup with dark background for better contrast ──
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Check if labels need Chinese font
    chinese_font = _try_chinese_font()
    font_props = {}
    if chinese_font:
        font_props = {'fontfamily': chinese_font}

    colors = EMOTION_COLORS[:len(label_names)]
    unique_labels = sorted(set(labels))

    # ── Layer 1: Confidence ellipses (2σ and 1σ) ──
    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        if mask.sum() < 3:
            continue
        color = colors[idx % len(colors)]
        ex = embeddings[mask, 0]
        ey = embeddings[mask, 1]
        
        # 2σ ellipse (outer, very light)
        _confidence_ellipse(
            ex, ey, ax, n_std=2.0,
            facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=1.0, linestyle='--'
        )
        # 1σ ellipse (inner, slightly more visible)
        _confidence_ellipse(
            ex, ey, ax, n_std=1.0,
            facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5, linestyle='-'
        )

    # ── Layer 2: KDE density contours (if scipy available) ──
    try:
        from scipy.stats import gaussian_kde
        for idx in unique_labels:
            if idx >= len(label_names):
                continue
            mask = labels == idx
            if mask.sum() < 10:
                continue
            color = colors[idx % len(colors)]
            ex = embeddings[mask, 0]
            ey = embeddings[mask, 1]
            
            try:
                kde = gaussian_kde(np.vstack([ex, ey]), bw_method=0.3)
                # Create grid
                x_grid = np.linspace(ex.min() - 0.3, ex.max() + 0.3, 80)
                y_grid = np.linspace(ey.min() - 0.3, ey.max() + 0.3, 80)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                
                # Draw contour lines only (no fill to keep it clean)
                ax.contour(X, Y, Z, levels=3, colors=[color], alpha=0.4, linewidths=0.8)
            except Exception:
                pass  # KDE can fail on degenerate data
    except ImportError:
        pass  # scipy not available, skip KDE

    # ── Layer 3: Scatter points ──
    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        color = colors[idx % len(colors)]
        n_points = mask.sum()
        
        # Adaptive point size based on sample count
        point_size = max(8, min(40, 2000 / max(len(features), 1)))
        
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=color,
            label=f'{label_names[idx]} ({n_points})',
            alpha=0.65,
            s=point_size,
            edgecolors='white',
            linewidths=0.3,
            zorder=5,
        )

    # ── Layer 4: Cluster centroids with label ──
    for idx in unique_labels:
        if idx >= len(label_names):
            continue
        mask = labels == idx
        if mask.sum() == 0:
            continue
        color = colors[idx % len(colors)]
        centroid = embeddings[mask].mean(axis=0)
        
        # Centroid marker
        ax.scatter(
            centroid[0], centroid[1],
            c=color, marker='D', s=120,
            edgecolors='black', linewidths=1.5,
            zorder=10
        )
        # Centroid label
        ax.annotate(
            label_names[idx],
            xy=(centroid[0], centroid[1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=9, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color, linewidth=0.5),
            zorder=11,
        )

    # ── Legend and styling ──
    legend = ax.legend(
        loc='upper right', fontsize=9, framealpha=0.9,
        fancybox=True, shadow=True,
        prop=font_props if font_props else None,
        title='Emotion (count)', title_fontsize=10,
        borderpad=0.8, labelspacing=0.6,
    )
    legend.get_frame().set_edgecolor('#CCCCCC')
    
    ax.set_title(f't-SNE Emotion Feature Clusters — {tag}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dim 1', fontsize=11, labelpad=8)
    ax.set_ylabel('t-SNE Dim 2', fontsize=11, labelpad=8)
    
    # Light grid
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='both', length=0)  # hide tick marks
    
    # Remove axis values (t-SNE dimensions are not meaningful)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#DDDDDD')
        spine.set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

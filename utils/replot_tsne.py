"""
Standalone re-plotting script for all analysis figures.

After a training run, analysis.py saves a *_tsne_features*.npz file
containing emotion features, labels, predicted labels, and (if available)
modality features (audio/video/fusion).

This script re-draws ALL plots from that saved npz file, so you can tweak
visualization parameters WITHOUT re-running training/evaluation.

Usage:
    # Re-plot everything (emotion t-SNE + modality t-SNE + confusion matrix):
    python utils/replot_tsne.py \
        --npz  /path/to/analysis/hmmem-qwen-meld-20260604/hmmem-qwen-meld-TEST-tsne_features.npz \
        --labels neutral surprise fear sadness joy disgust anger

    # Only emotion t-SNE with all samples (no balanced sampling):
    python utils/replot_tsne.py --npz /path/to/file.npz --max_per_class 0

    # Only confusion matrix:
    python utils/replot_tsne.py --npz /path/to/file.npz --only confusion_matrix

All arguments except --npz are optional (auto-inferred when possible).
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path so we can import utils.analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.analysis import (
    _plot_emotion_tsne, _plot_confusion_matrix, _plot_modality_tsne,
    _save_multi_format, logger,
)


def main():
    parser = argparse.ArgumentParser(
        description='Re-plot t-SNE / Confusion Matrix / Modality t-SNE from saved npz features'
    )
    parser.add_argument('--npz', type=str, required=True,
                        help='Path to *_tsne_features*.npz saved by analysis.py')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Emotion class names in index order. '
                             'E.g.: neutral surprise fear sadness joy disgust anger')
    parser.add_argument('--tag', type=str, default=None,
                        help='Experiment tag for plot title (auto-inferred from filename if omitted)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (defaults to same dir as npz file)')
    parser.add_argument('--max_per_class', type=int, default=150,
                        help='Max samples per emotion class for balanced t-SNE. '
                             '0 = use all samples (default: 150)')
    parser.add_argument('--only', type=str, default=None,
                        choices=['emotion', 'modality', 'confusion_matrix'],
                        help='Only re-plot this specific figure type. '
                             'Default: re-plot all available figures.')

    args = parser.parse_args()

    # ── Load npz ──
    if not os.path.exists(args.npz):
        print(f"ERROR: npz file not found: {args.npz}")
        sys.exit(1)

    data = np.load(args.npz)
    features = data['features']
    true_labels = data['labels']
    pred_labels = data['pred_labels']

    # Check for modality features
    has_modality = any(f'mod_{m}' in data for m in ('audio', 'video', 'fusion'))
    modality_features = {}
    if has_modality:
        for mod_name in ('audio', 'video', 'fusion'):
            key = f'mod_{mod_name}'
            if key in data:
                modality_features[mod_name] = data[key]

    print(f"Loaded: features={features.shape}, "
          f"true_labels={true_labels.shape}, pred_labels={pred_labels.shape}")
    print(f"Unique true labels: {np.unique(true_labels)}")
    print(f"Unique pred labels: {np.unique(pred_labels)}")
    if modality_features:
        mod_info = ', '.join(f'{k}={v.shape}' for k, v in modality_features.items())
        print(f"Modality features: {mod_info}")
    else:
        print("Modality features: not available in npz (older format)")

    # ── Auto-infer parameters ──
    out_dir = args.out or os.path.dirname(args.npz)
    os.makedirs(out_dir, exist_ok=True)

    # Tag: infer from filename  e.g. "hmmem-qwen-meld-TEST-tsne_features.npz"
    if args.tag is None:
        basename = os.path.basename(args.npz)
        tag = basename.split('-tsne_features')[0] if '-tsne_features' in basename else 'replot'
    else:
        tag = args.tag

    # Labels: auto-generate if not provided
    if args.labels is None:
        n_classes = int(true_labels.max()) + 1
        label_names = [f'class_{i}' for i in range(n_classes)]
        print(f"WARNING: --labels not provided, using generic names: {label_names}")
        print(f"         Consider specifying: --labels neutral surprise fear ...")
    else:
        label_names = args.labels

    only = args.only

    # ── Re-plot Emotion t-SNE ──
    if only is None or only == 'emotion':
        print(f"\nRe-plotting Emotion t-SNE (max_per_class={args.max_per_class})...")
        tsne_path = os.path.join(out_dir, f'{tag}-tsne_emotion_true-replot.png')
        _plot_emotion_tsne(
            features, true_labels=true_labels, pred_labels=pred_labels,
            label_names=label_names, save_path=tsne_path, tag=tag,
            max_per_class=args.max_per_class,
        )
        print(f"  → True labels:  {tsne_path} (+.svg)")
        print(f"  → Pred labels:  {tsne_path.replace('_true', '_pred')} (+.svg)")
        print(f"  → Correct only: {tsne_path.replace('_true', '_correct')} (+.svg)")

    # ── Re-plot Modality t-SNE ──
    if only is None or only == 'modality':
        if modality_features:
            print(f"\nRe-plotting Modality t-SNE...")
            mod_path = os.path.join(out_dir, f'{tag}-tsne_modality-replot.png')
            _plot_modality_tsne(modality_features, mod_path, tag)
            print(f"  → {mod_path} (+.svg)")
        else:
            print("\n⚠️  Skipping Modality t-SNE: no modality features in npz file.")
            print("   (This npz was saved by an older version. Re-run the experiment to include them.)")

    # ── Re-plot Confusion Matrix ──
    if only is None or only == 'confusion_matrix':
        print(f"\nRe-plotting Confusion Matrix...")
        cm_path = os.path.join(out_dir, f'{tag}-confusion_matrix-replot.png')
        _plot_confusion_matrix(true_labels, pred_labels, label_names, cm_path, tag)
        print(f"  → {cm_path} (+.svg)")

    print("\nDone!")


if __name__ == '__main__':
    main()

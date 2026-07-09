"""
Standalone re-plotting script for all analysis figures.

FAST MODE (default): If the npz contains cached t-SNE embeddings (saved by
the latest analysis.py), replotting is instant — no t-SNE recomputation.

RECOMPUTE MODE (--recompute): Force re-running t-SNE from raw features.
Use this when you changed preprocessing/t-SNE hyperparameters.

Usage:
    # Re-plot everything FAST from cached embeddings:
    python utils/replot_tsne.py \
        --npz  /path/to/hmmem-qwen-meld-TEST-tsne_features.npz \
        --labels neutral surprise fear sadness joy disgust anger

    # Force re-run t-SNE (slow but applies new hyperparams):
    python utils/replot_tsne.py --npz /path/to/file.npz --recompute

    # Only emotion t-SNE, all samples:
    python utils/replot_tsne.py --npz /path/to/file.npz --max_per_class 0

    # Only confusion matrix:
    python utils/replot_tsne.py --npz /path/to/file.npz --only confusion_matrix

All arguments except --npz are optional (auto-inferred when possible).
"""

import argparse
import os
import sys
import time
import numpy as np

# Add project root to path so we can import utils.analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.analysis import (
    _plot_emotion_tsne, _plot_confusion_matrix, _plot_confusion_matrix_clean,
    _plot_modality_tsne, _draw_tsne_scatter, _draw_modality_scatter,
    _save_multi_format, logger,
    MODALITY_COLORS, MODALITY_LABELS,
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
    parser.add_argument('--recompute', action='store_true',
                        help='Force re-running t-SNE from raw features '
                             '(slow, use when hyperparams changed)')

    args = parser.parse_args()

    # ── Load npz ──
    if not os.path.exists(args.npz):
        print(f"ERROR: npz file not found: {args.npz}")
        sys.exit(1)

    data = np.load(args.npz)
    features = data['features']
    true_labels = data['labels']
    pred_labels = data['pred_labels']

    # Check for cached embeddings
    has_emo_cache = 'emo_embeddings' in data
    has_mod_cache = 'mod_embeddings' in data

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
    print(f"Cached emotion embeddings: {'yes' if has_emo_cache else 'no'}")
    print(f"Cached modality embeddings: {'yes' if has_mod_cache else 'no'}")
    if modality_features:
        mod_info = ', '.join(f'{k}={v.shape}' for k, v in modality_features.items())
        print(f"Modality features: {mod_info}")
    if args.recompute:
        print("⚠️  --recompute: will re-run t-SNE from raw features (slow)")

    # ── Auto-infer parameters ──
    out_dir = args.out or os.path.dirname(args.npz)
    os.makedirs(out_dir, exist_ok=True)

    if args.tag is None:
        basename = os.path.basename(args.npz)
        tag = basename.split('-tsne_features')[0] if '-tsne_features' in basename else 'replot'
    else:
        tag = args.tag

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
        t0 = time.time()
        tsne_path = os.path.join(out_dir, f'{tag}-tsne_emotion_true-replot.png')

        if has_emo_cache and not args.recompute:
            # FAST: use cached embeddings, skip t-SNE computation
            print(f"\nRe-plotting Emotion t-SNE from CACHED embeddings (instant)...")
            emo_emb = data['emo_embeddings']
            emo_tl = data['emo_true_labels']
            emo_pl = data['emo_pred_labels']

            _draw_tsne_scatter(emo_emb, emo_tl, label_names,
                               tsne_path, tag, title_suffix='True Labels')

            pred_path = tsne_path.replace('_true', '_pred')
            _draw_tsne_scatter(emo_emb, emo_pl, label_names,
                               pred_path, tag, title_suffix='Predicted Labels')

            correct_mask = emo_tl == emo_pl
            if correct_mask.sum() >= 10:
                correct_path = tsne_path.replace('_true', '_correct')
                _draw_tsne_scatter(emo_emb[correct_mask], emo_tl[correct_mask],
                                   label_names, correct_path, tag,
                                   title_suffix='Correct Predictions Only')
        else:
            # SLOW: recompute t-SNE from raw features
            print(f"\nRe-plotting Emotion t-SNE (RECOMPUTING, max_per_class={args.max_per_class})...")
            _plot_emotion_tsne(
                features, true_labels=true_labels, pred_labels=pred_labels,
                label_names=label_names, save_path=tsne_path, tag=tag,
                max_per_class=args.max_per_class,
            )

        elapsed = time.time() - t0
        print(f"  → True labels:  {tsne_path} (+.svg)")
        print(f"  → Pred labels:  {tsne_path.replace('_true', '_pred')} (+.svg)")
        print(f"  → Correct only: {tsne_path.replace('_true', '_correct')} (+.svg)")
        print(f"  ⏱ {elapsed:.1f}s")

    # ── Re-plot Modality t-SNE ──
    if only is None or only == 'modality':
        t0 = time.time()
        mod_path = os.path.join(out_dir, f'{tag}-tsne_modality-replot.png')

        if has_mod_cache and not args.recompute:
            # FAST: use cached embeddings
            print(f"\nRe-plotting Modality t-SNE from CACHED embeddings (instant)...")
            mod_emb = data['mod_embeddings']
            mod_lbl = data['mod_labels']
            # Reconstruct modality_counts from labels
            modality_order = ['fusion', 'audio', 'video']
            modality_counts = {}
            for idx, name in enumerate(modality_order):
                count = int((mod_lbl == idx).sum())
                if count > 0:
                    modality_counts[name] = count
            _draw_modality_scatter(mod_emb, mod_lbl, modality_order,
                                   modality_counts, mod_path)
            elapsed = time.time() - t0
            print(f"  → {mod_path} (+.svg)")
            print(f"  ⏱ {elapsed:.1f}s")
        elif modality_features:
            # SLOW: recompute
            print(f"\nRe-plotting Modality t-SNE (RECOMPUTING)...")
            _plot_modality_tsne(modality_features, mod_path, tag)
            elapsed = time.time() - t0
            print(f"  → {mod_path} (+.svg)")
            print(f"  ⏱ {elapsed:.1f}s")
        else:
            print("\n⚠️  Skipping Modality t-SNE: no modality data in npz file.")
            print("   (Re-run the experiment to include modality features.)")

    # ── Re-plot Confusion Matrix ──
    if only is None or only == 'confusion_matrix':
        print(f"\nRe-plotting Confusion Matrix...")
        cm_path = os.path.join(out_dir, f'{tag}-confusion_matrix-replot.png')
        _plot_confusion_matrix(true_labels, pred_labels, label_names, cm_path, tag)
        print(f"  → {cm_path} (+.svg)")

        cm_clean_path = os.path.join(out_dir, f'{tag}-confusion_matrix_clean-replot.png')
        _plot_confusion_matrix_clean(true_labels, pred_labels, label_names, cm_clean_path, tag)
        print(f"  → {cm_clean_path} (+.svg) (clean/paper version)")

    print("\nDone!")


if __name__ == '__main__':
    main()

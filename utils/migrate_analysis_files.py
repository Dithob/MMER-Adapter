"""
One-time migration script: reorganize legacy flat analysis files into
per-experiment subdirectories, AND update CSV result files with new paths.

Before:
  analysis/
  ├── hmmem-chatglm3-meld-TEST-confusion_matrix-20260604_102603.png
  ├── hmmem-chatglm3-meld-TEST-tsne_emotion-20260604_102603.png
  └── ...

After:
  analysis/
  ├── hmmem-chatglm3-meld-20260604_102603/
  │   ├── hmmem-chatglm3-meld-TEST-confusion_matrix.png
  │   └── hmmem-chatglm3-meld-TEST-tsne_emotion.png
  └── ...

CSV files in the results directory will have their ConfusionMatrix / tSNE
path columns updated to point to the new locations.

Usage:
    # Dry-run first (default) to see what would happen:
    python utils/migrate_analysis_files.py /path/to/analysis/

    # Actually move files AND update CSVs:
    python utils/migrate_analysis_files.py /path/to/analysis/ --execute

    # Specify a different CSV directory (default: parent of analysis_dir):
    python utils/migrate_analysis_files.py /path/to/analysis/ --execute --csv_dir /path/to/results/
"""

import os
import re
import sys
import shutil
import argparse
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Pattern: {tag}-{plot_type}-{timestamp}.{ext}
# e.g.: hmmem-chatglm3-meld-TEST-confusion_matrix-20260604_102603.png
# timestamp: YYYYMMDD_HHMMSS (always 15 chars: 8+1+6)
TIMESTAMP_RE = re.compile(r'-(\d{8}_\d{6})\.(png|svg|pdf|npz)$')


def parse_file(filename):
    """Parse a legacy analysis filename into components.

    Returns:
        (experiment_folder, new_filename, timestamp) or None if not parseable.
    """
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return None

    timestamp = match.group(1)
    ext = match.group(2)

    # Remove timestamp from filename to get: {tag}-{plot_type}.{ext}
    new_filename = filename[:match.start()] + f'.{ext}'

    # Extract experiment folder: {modelName}-{model_type}-{datasetName}-{timestamp}
    # The base_no_ext is like: hmmem-chatglm3-meld-TEST-confusion_matrix
    # We need to find the MODE part (TEST or VALID) and extract everything before it
    base_no_ext = filename[:match.start()]

    # Find MODE (TEST or VALID or VAL)
    mode_match = re.search(r'-(TEST|VALID|VAL)-', base_no_ext)
    if mode_match:
        experiment_base = base_no_ext[:mode_match.start()]
        folder = f'{experiment_base}-{timestamp}'
    else:
        folder = f'unknown-{timestamp}'

    return folder, new_filename, timestamp


def find_csv_files(csv_dir):
    """Find all CSV files in the given directory."""
    if not os.path.isdir(csv_dir):
        return []
    return [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
            if f.endswith('.csv') and os.path.isfile(os.path.join(csv_dir, f))]


def update_csv_paths(csv_files, path_mapping, dry_run=True):
    """Update ConfusionMatrix and tSNE columns in CSV files.

    Args:
        csv_files: list of CSV file paths
        path_mapping: dict mapping old_path → new_path
        dry_run: if True, only report what would change
    
    Returns:
        total number of path updates made
    """
    if not HAS_PANDAS:
        print("⚠️  pandas not available, skipping CSV updates.")
        print("   Install pandas to enable CSV path migration.")
        return 0

    # Columns that contain analysis file paths
    path_columns = ['ConfusionMatrix', 'tSNE']
    total_updates = 0

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            print(f"  ⚠️  Failed to read {csv_path}: {e}")
            continue

        csv_updates = 0
        for col in path_columns:
            if col not in df.columns:
                continue
            for idx, val in df[col].items():
                if pd.isna(val) or val == '':
                    continue
                # Check if this path matches any in our mapping
                val_str = str(val).strip()
                if val_str in path_mapping:
                    new_val = path_mapping[val_str]
                    df.at[idx, col] = new_val
                    csv_updates += 1

        if csv_updates > 0:
            csv_name = os.path.basename(csv_path)
            print(f"  📝 {csv_name}: {csv_updates} path(s) updated")
            if not dry_run:
                df.to_csv(csv_path, index=False)
            total_updates += csv_updates

    return total_updates


def main():
    parser = argparse.ArgumentParser(
        description='Migrate flat analysis files into per-experiment subdirectories '
                    'and update CSV result paths'
    )
    parser.add_argument('analysis_dir', type=str,
                        help='Path to the analysis/ directory')
    parser.add_argument('--execute', action='store_true',
                        help='Actually move files and update CSVs (default is dry-run)')
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='Directory containing CSV result files '
                             '(default: parent directory of analysis_dir)')

    args = parser.parse_args()

    if not os.path.isdir(args.analysis_dir):
        print(f"ERROR: {args.analysis_dir} is not a directory")
        sys.exit(1)

    # CSV directory: default to parent of analysis/ (i.e. results/)
    csv_dir = args.csv_dir or os.path.dirname(os.path.abspath(args.analysis_dir))

    # Collect all files in the flat directory (not in subdirs)
    files = [f for f in os.listdir(args.analysis_dir)
             if os.path.isfile(os.path.join(args.analysis_dir, f))]

    if not files:
        print("No files found in the analysis directory.")
        return

    # Parse and group
    moves = []  # (old_path, new_path, folder)
    path_mapping = {}  # old_abs_path → new_abs_path (for CSV updates)
    skipped = []

    for filename in sorted(files):
        result = parse_file(filename)
        if result is None:
            skipped.append(filename)
            continue

        folder, new_filename, timestamp = result
        old_path = os.path.join(args.analysis_dir, filename)
        new_dir = os.path.join(args.analysis_dir, folder)
        new_path = os.path.join(new_dir, new_filename)
        moves.append((old_path, new_path, folder))

        # Build path mapping for CSV updates (use absolute paths)
        old_abs = os.path.abspath(old_path)
        new_abs = os.path.abspath(new_path)
        path_mapping[old_abs] = new_abs
        # Also map without abspath in case CSV has relative or different prefix
        path_mapping[old_path] = new_path

    # Group by folder for display
    by_folder = defaultdict(list)
    for old, new, folder in moves:
        by_folder[folder].append((os.path.basename(old), os.path.basename(new)))

    # Find CSV files
    csv_files = find_csv_files(csv_dir)

    # Display plan
    mode_label = '(EXECUTING)' if args.execute else '(DRY RUN)'
    print(f"{'=' * 60}")
    print(f"  Analysis File Migration {mode_label}")
    print(f"{'=' * 60}")
    print(f"  Source:       {args.analysis_dir}")
    print(f"  CSV dir:      {csv_dir}")
    print(f"  Files found:  {len(files)}")
    print(f"  Files to move: {len(moves)}")
    print(f"  Skipped:      {len(skipped)} (no timestamp pattern)")
    print(f"  Target folders: {len(by_folder)}")
    print(f"  CSV files:    {len(csv_files)}")
    print(f"{'=' * 60}\n")

    # Show file moves
    print("── File Moves ──\n")
    for folder in sorted(by_folder.keys()):
        file_pairs = by_folder[folder]
        print(f"📁 {folder}/")
        for old_name, new_name in file_pairs:
            if old_name != new_name:
                print(f"    {old_name}  →  {new_name}")
            else:
                print(f"    {old_name}")
        print()

    if skipped:
        print(f"⚠️  Skipped files (no timestamp pattern):")
        for s in skipped:
            print(f"    {s}")
        print()

    # Show CSV updates
    if csv_files:
        print("── CSV Path Updates ──\n")
        csv_update_count = update_csv_paths(csv_files, path_mapping, dry_run=not args.execute)
        if csv_update_count == 0:
            print("  (no matching paths found in CSVs)")
        print()

    # Execute moves
    if args.execute:
        moved = 0
        for old_path, new_path, folder in moves:
            new_dir = os.path.dirname(new_path)
            os.makedirs(new_dir, exist_ok=True)
            shutil.move(old_path, new_path)
            moved += 1
        print(f"✅ Moved {moved} files into {len(by_folder)} folders.")
        if csv_files and HAS_PANDAS:
            print(f"✅ CSV paths updated ({csv_update_count} entries).")
    else:
        print("This is a DRY RUN. No files were moved, no CSVs were modified.")
        print("Run with --execute to actually perform the migration.")


if __name__ == '__main__':
    main()

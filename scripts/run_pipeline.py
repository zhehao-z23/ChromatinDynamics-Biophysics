"""Run the frozen v5.2.1 physical workflow without changing scientific parameters."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('qc', 'physics', 'selection', 'msd', 'vac', 'vcc', 'display',
          'separation', 'mscd-clean', 'vac-best-worst')


def validate_path_separation(cache: Path, output: Path, snapshot: Path | None = None) -> None:
    """Keep generated results out of every frozen input tree, including via symlinks."""
    paths = [('cache', cache.resolve()), ('output-root', output.resolve())]
    if snapshot is not None:
        paths.append(('snapshot', snapshot.resolve()))
    for index, (name, path) in enumerate(paths):
        for other_name, other_path in paths[index + 1:]:
            if path == other_path or path in other_path.parents or other_path in path.parents:
                raise ValueError(f'{name} and {other_name} must be separate, non-overlapping directories')


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    out, cache = args.output_root.resolve(), args.cache.resolve()
    ref = ROOT / 'references'
    cfg = args.config.resolve()
    fits = out / 'physics/tables/fits.csv'
    selection = out / 'selection/tables/eligible_complete_pair_trajectory_summary.csv'
    commands = []
    def add(name: str, script: str, *items: object) -> None:
        commands.append((name, [sys.executable, str(ROOT / 'scripts' / script),
                                *[str(item) for item in items]]))
    if args.snapshot:
        add('cache', '21_build_v521_unfiltered_cache.py', '--snapshot', args.snapshot.resolve(),
            '--output-dir', cache)
    definitions = {
        'qc': ('22_build_v521_pair_galleries.py', '--cache', cache, '--output-dir', out/'qc'),
        'physics': ('23_build_v521_time_resolved_physics.py', '--cache', cache, '--config', cfg,
                    '--output-dir', out/'physics'),
        'selection': ('26_select_v521_complete_pair_cases_without_t5_t10.py', '--cache', cache,
                      '--output-dir', out/'selection'),
        'msd': ('37_build_v521_final_msd_visualization.py', '--physics-run', out/'physics',
                '--complete-pair-csv', selection, '--config', ROOT/'config/v521_final_msd_visualization.json',
                '--output-dir', out/'msd'),
        'vac': ('38_build_v521_oligo_vac.py', '--cache', cache, '--config', cfg,
                '--msd-fits', fits, '--reference-vvcf-dir', ref/'rouse',
                '--reference-readme-pdf', ref/'Read_Me.pdf', '--output-dir', out/'vac'),
        'vcc': ('39_build_v521_oligo_vcc.py', '--cache', cache, '--config', cfg,
                '--empirical-reference-dir', ref/'empirical', '--vvcf-reference-dir', ref/'rouse',
                '--reference-readme-pdf', ref/'Read_Me.pdf', '--output-dir', out/'vcc'),
        'display': ('40_build_v521_vac_vcc_visual_candidates.py', '--vac-result', out/'vac',
                    '--vcc-result', out/'vcc', '--msd-fits', fits, '--output-dir', out/'display'),
        'separation': ('42_build_v521_separation_final_ppt.py', '--cache', cache,
                       '--threshold-nm', '500', '--multiple-testing', 'bonferroni',
                       '--output-dir', out/'separation'),
        'mscd-clean': ('46_build_v521_mscd10_macro_time_clean.py', '--fits-table', fits,
                       '--output-dir', out/'mscd-clean'),
        'vac-best-worst': ('47_build_v521_vac_fbm_best_worst_panel.py',
                           '--dense-vac-table', out/'display/tables/vac_fine_display_means.csv',
                           '--consistency-table', out/'vac/tables/vac_msd_consistency.csv',
                           '--output-dir', out/'vac-best-worst'),
    }
    for name in STAGES:
        if name in args.stages: add(name, *definitions[name])
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path,
                        help='Existing frozen flat cache, or a NEW path when --snapshot is set.')
    parser.add_argument('--snapshot', type=Path,
                        help='Optional frozen intake to build a new cache first; existing cache is never overwritten.')
    parser.add_argument('--output-root', required=True, type=Path,
                        help='New analysis root. For partial resumption use only uncompleted --stages.')
    parser.add_argument('--config', type=Path, default=ROOT/'config/physics.yaml')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--dry-run', action='store_true', help='Print command plan only; create no files.')
    args = parser.parse_args()
    try:
        validate_path_separation(args.cache, args.output_root, args.snapshot)
    except ValueError as exc:
        parser.error(str(exc))
    commands = build_commands(args)
    print(json.dumps([{'stage': n, 'command': c} for n,c in commands], indent=2))
    if args.dry_run: return 0
    if not args.config.is_file(): parser.error(f'Configuration not found: {args.config}')
    if args.snapshot and args.cache.exists(): parser.error('--snapshot requires a new --cache path.')
    if not args.snapshot and not (args.cache/'CACHE_CONTRACT.json').is_file():
        parser.error('--cache must contain CACHE_CONTRACT.json; plain trajectory CSVs are insufficient.')
    for name, _ in commands:
        if name != 'cache' and (args.output_root/name).exists():
            parser.error(f'Stage output already exists: {args.output_root/name}; use a new root or --stages.')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('MPLBACKEND', 'Agg')
    for name, command in commands:
        print(f'Running {name}', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay every released route and recompute paired results using Python 3.10+."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fscr.scoring import ANNOTATED, DIALOGUE_DERIVED, route, route_from_scores


def read_csv(name):
    with (ROOT / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def read_json(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f'Numeric mismatch: {actual} != {expected}')


def fired(d, b, params=ANNOTATED):
    return route_from_scores(float(d), float(b), params)['selected_system'] == 'base'


def mcnemar(fixes, regressions):
    n = fixes + regressions
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(fixes, regressions) + 1)) / 2**n)


def metrics(rows, switches):
    require(len(rows) == len(switches) and bool(rows), 'Invalid metric input')
    fixes = sum(s and int(r['base_label']) == 1 and int(r['distill_label']) == 0 for r, s in zip(rows, switches))
    regressions = sum(s and int(r['base_label']) == 0 and int(r['distill_label']) == 1 for r, s in zip(rows, switches))
    n = len(rows)
    return dict(n=n, routes=sum(switches), fixes=fixes, regressions=regressions,
                gain_pp=100 * (fixes - regressions) / n,
                precision=fixes / (fixes + regressions) if fixes + regressions else None,
                mcnemar_p=mcnemar(fixes, regressions))


def main(args):
    if (ROOT / 'MANIFEST.json').exists():
        manifest = read_json('MANIFEST.json')
        for name, metadata in manifest['files'].items():
            content = (ROOT / name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == metadata['sha256'], f'Checksum mismatch: {name}')
        print(f"Verified {len(manifest['files'])} file checksums.\n")

    primary = read_csv('data/main.csv')
    reference = read_json('results/reference_main.json')
    grouped = defaultdict(list)
    primary_index = {}
    for row in primary:
        key = row['benchmark'], row['pair'], row['eval_id']
        require(key not in primary_index, f'Duplicate example: {key}')
        primary_index[key] = row
        switch = fired(row['distill_leak'], row['base_leak'])
        require(switch == bool(int(row['triggered'])), f'Route mismatch: {key}')
        for side in ['base', 'distill']:
            votes = row[f'{side}_votes']
            require(bool(votes) == switch, f'Vote coverage mismatch: {key}')
            if switch:
                require(len(votes) == 3 and set(votes) <= {'0', '1'}, f'Malformed votes: {key}')
                require(int(row[f'{side}_label']) == int(votes.count('1') >= 2), f'Majority mismatch: {key}')
        grouped[row['benchmark'], row['pair']].append(row)

    report = {'primary': {}, 'state_inputs': {}, 'mquake': {}}
    print('| Benchmark / split | Pair | N | Routes | Fix/reg. | Gain (pp) |')
    print('|---|---|---:|---:|---:|---:|')
    for (benchmark, pair), rows in grouped.items():
        expected_n = 2982 if benchmark == 'icf_dsm' else 994
        require(len(rows) == expected_n, f'Incomplete population: {benchmark}/{pair}')
        require(len({r['source_id'] for r in rows}) == 994, 'Dialogue population mismatch')
        for split in ['full', 'tail'] if benchmark == 'icf_dsm' else ['full']:
            selected = rows if split == 'full' else [r for r in rows if int(r['source_id']) >= 562]
            if split == 'tail':
                require(len(selected) == 1302 and len({r['source_id'] for r in selected}) == 434, 'Invalid held-out tail')
            result = metrics(selected, [bool(int(r['triggered'])) for r in selected])
            ref = reference[benchmark][pair]['triggered_primary' if split == 'full' else 'fresh_tail_triggered']
            for key in ['fixes', 'regressions']:
                require(result[key] == ref[key], f'Reference mismatch: {benchmark}/{pair}/{key}')
            require(result['routes'] == ref['n'], 'Reference route count mismatch')
            close(result['mcnemar_p'], ref['mcnemar_exact_p'])
            report['primary'][f'{benchmark}/{pair}/{split}'] = result
            print(f"| {benchmark}/{split} | {pair} | {result['n']} | {result['routes']} | {result['fixes']}/{result['regressions']} | {result['gain_pp']:+.2f} |")
        triggered = [r for r in rows if int(r['triggered'])]
        for side in ['base', 'distill']:
            ref = reference[benchmark][pair]['agreement'][side]
            require(sum(len(set(r[f'{side}_votes'])) == 1 for r in triggered) == ref['unanimous'], 'Vote agreement mismatch')
    require(len(grouped) == 8, 'Expected four pairs on each primary benchmark')

    example_count = 0
    for line in (ROOT / 'data/examples.jsonl').open():
        row = json.loads(line)
        prediction = route(row['base_answer'], row['distill_answer'], row['forgotten'])
        reference_row = primary_index[row['benchmark'], row['pair'], row['eval_id']]
        for key in ['base_leak', 'distill_leak']:
            close(prediction[key], float(reference_row[key]))
        example_count += 1
    print(f'\nVerified exact lexical scores on {example_count} deterministic text fixtures.')

    state_grouped = defaultdict(list)
    for row in read_csv('data/state_inputs.csv'):
        state_grouped[row['pair']].append(row)
    state_reference = {(r['pair'], r['method']): r for r in read_csv('results/reference_state_inputs.csv')}
    for pair, rows in state_grouped.items():
        require(len(rows) == 1302 and len({r['eval_id'] for r in rows}) == 1302, 'Incomplete state-input population')
        rules = {
            'gold_fscr': lambda r: fired(r['gold_distill_leak'], r['gold_base_leak']),
            'source_surface_fscr': lambda r: fired(r['source_surface_distill_leak'], r['source_surface_base_leak']),
            'predicted_fscr': lambda r: fired(r['predicted_distill_leak'], r['predicted_base_leak'], DIALOGUE_DERIVED),
            'gold_lower_leak': lambda r: float(r['gold_base_leak']) < float(r['gold_distill_leak']),
            'gold_distill_high_only': lambda r: float(r['gold_distill_leak']) >= .5,
            'gold_margin_only': lambda r: float(r['gold_distill_leak']) - float(r['gold_base_leak']) >= .15,
            'random_same_gold_coverage': lambda r: bool(int(r['random_route'])),
        }
        for method, rule in rules.items():
            result = metrics(rows, [rule(r) for r in rows])
            ref = state_reference[pair, method]
            for key in ['n', 'fixes', 'regressions']:
                require(result[key] == int(ref[key]), f'State mismatch: {pair}/{method}/{key}')
            close(result['routes'] / result['n'], float(ref['coverage']))
            close(result['gain_pp'], float(ref['net_pp']))
            report['state_inputs'][f'{pair}/{method}'] = result
    print('Verified all 28 state-input/control results against the stored paper aggregates.')

    external = defaultdict(list)
    for row in read_csv('data/mquake_scores.csv'):
        require(fired(row['distill_leak'], row['base_leak']) == bool(int(row['triggered'])), 'MQuAKE route mismatch')
        external[row['pair']].append(row)
    # Exact counts in the current manuscript's MQuAKE table (not fitted values).
    expected_external = {'Qwen-1.5B': (1000, 46, 0), 'Qwen-7B': (999, 10, 0),
        'Qwen-14B': (1000, 1, 0), 'Llama-8B': (999, 9, 0),
        'OpenThinker3-7B': (998, 44, 19), 'Falcon-H1R-7B': (1000, 1, 0)}
    for pair, rows in external.items():
        require(len({r['eval_id'] for r in rows}) == len(rows), 'Duplicate MQuAKE IDs')
        result = metrics(rows, [bool(int(r['triggered'])) for r in rows])
        require((result['n'], result['fixes'], result['regressions']) == expected_external[pair], f'MQuAKE paper mismatch: {pair}')
        report['mquake'][pair] = result
    require(set(external) == set(expected_external), 'Missing external model pairs')
    print('Verified all six MQuAKE effects, including regressions and incomplete label intersections.')

    if args.clustered:
        from clustered import summarize
        reference_cluster = read_json('results/reference_clustered_inference.json')
        report['clustered'] = {}
        for index, pair in enumerate(sorted(reference_cluster)):
            rows = grouped['icf_dsm', pair]
            contributions = defaultdict(int)
            for row in rows:
                contributions[int(row['source_id'])] += int(row['triggered']) * (int(row['base_label']) - int(row['distill_label']))
            ids = sorted(contributions)
            for split, selected, seed in [('full', ids, 20260808 + 10 * index),
                    ('fresh_562plus', [i for i in ids if i >= 562], 20261808 + 10 * index)]:
                result = summarize(contributions, selected, seed=seed)
                ref = reference_cluster[pair][split]
                require(result == ref, f'Clustered inference mismatch: {pair}/{split}')
                report['clustered'][f'{pair}/{split}'] = result
                print(f'Verified clustered inference: {pair}/{split}', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('\nAll requested checks passed. No model generation or rejudging was performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clustered', action='store_true', help='Also recompute the original seeded bootstrap and sign-flip tests (several CPU minutes).')
    parser.add_argument('--output', type=Path, help='Write recomputed metrics to this JSON file.')
    main(parser.parse_args())

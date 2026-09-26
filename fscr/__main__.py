"""Route JSONL candidate pairs: python -m fscr < candidates.jsonl."""
import argparse
import json
import sys

from .scoring import ANNOTATED, DIALOGUE_DERIVED, route
from .state import dialogue_only_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', choices=['annotated', 'dialogue'], default='annotated')
    args = parser.parse_args()
    for number, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if args.state == 'dialogue':
                forgotten = dialogue_only_state(row['conversation'], row['target_turn'])['spans']
                thresholds = DIALOGUE_DERIVED
            else:
                forgotten = row['forgotten']
                thresholds = ANNOTATED
            if not isinstance(forgotten, list) or not all(isinstance(s, str) for s in forgotten):
                raise ValueError('forgotten must be a list of strings')
            if not all(isinstance(row[k], str) for k in ['base_answer', 'distill_answer']):
                raise ValueError('candidate answers must be strings')
            result = route(row['base_answer'], row['distill_answer'], forgotten, thresholds)
            print(json.dumps({'eval_id': row.get('eval_id'), **result}, ensure_ascii=False))
        except (KeyError, ValueError, TypeError) as error:
            raise SystemExit(f'Input line {number}: {error}') from error


if __name__ == '__main__':
    main()

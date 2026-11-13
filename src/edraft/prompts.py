"""Deterministic, dataset-disjoint benchmark prompt manifests."""
from __future__ import annotations

import json
from pathlib import Path

from .runtime import ROOT, atomic_json, common_parser, fingerprint, parse_config, record_run

DATASETS = ['openai/gsm8k', 'openai/openai_humaneval', 'HuggingFaceH4/MATH-500']


def local_rows(path, subset=None, split=None):
    import pyarrow.parquet as pq
    paths = sorted(Path(path).rglob('*.parquet'))
    if subset:
        paths = [p for p in paths if subset in p.parts]
    if split:
        paths = [p for p in paths if split in p.name or split in p.parts]
    if not paths:
        jsonl = sorted(Path(path).rglob('*.jsonl'))
        if subset:
            jsonl = [p for p in jsonl if subset in p.parts]
        if split:
            jsonl = [p for p in jsonl if split in p.name or split in p.parts]
        if jsonl:
            return [json.loads(line) for p in jsonl for line in p.read_text().splitlines() if line.strip()]
        raise RuntimeError(f'blocked: no parquet/JSONL files for subset={subset}, split={split} in {path}')
    return [row for p in paths for row in pq.read_table(p).to_pylist()]


def stable_order(items, seed):
    return sorted(items, key=lambda x: fingerprint({'seed': seed, 'id': x['prompt_id']}))


def build_manifest(args):
    from .download import fetch_dataset
    record_run(args, 'prompts')
    if args.scale == 'smoke':
        rows = [{'prompt_id': f'synthetic-{i}', 'dataset': 'builtin', 'split': split,
                 'condition': 'standard_nonthinking', 'text': f'Synthetic prompt {i}'}
                for i, split in enumerate(['train', 'dev', 'test'])]
        result = {'scale': 'smoke', 'seed': args.seed, 'prompts': rows, 'calibration_ids': ['synthetic-0']}
        atomic_json(args.output / 'prompts-smoke.json', result)
        return result
    locations = {dataset: fetch_dataset(dataset, args.allow_download) for dataset in DATASETS}
    env = json.loads((ROOT / 'artifacts/environment.json').read_text())
    gsm = local_rows(locations[DATASETS[0]], subset='main', split='train')
    standard = []
    for index, row in enumerate(gsm):
        standard.append({'prompt_id': f'gsm8k-main-train-{index:05d}', 'dataset': DATASETS[0],
                         'source_row': index, 'condition': 'standard_nonthinking',
                         'text': row['question'] + '\nExplain the reasoning and give the final answer.'})
    ordered = stable_order(standard, args.seed)
    dev_count = 64
    for i, row in enumerate(ordered):
        row['split'] = 'dev' if i < dev_count else 'train'
    train = [row for row in ordered if row['split'] == 'train']
    calibration = [row['prompt_id'] for row in train[:64]]
    humaneval = local_rows(locations[DATASETS[1]], split='test')
    heldout = []
    for index, row in enumerate(humaneval):
        heldout.append({'prompt_id': 'humaneval-' + row['task_id'].replace('/', '-'),
                        'dataset': DATASETS[1], 'source_row': index, 'split': 'test',
                        'condition': 'standard_nonthinking',
                        'text': 'Write a correct Python solution to this programming task. Include the completed '
                                'function and a concise explanation.\n```python\n' + row['prompt'] + '\n```'})
    # Preserve the complete dataset as held-out. A fixed first 64 form reference evaluation.
    heldout = stable_order(heldout, args.seed)
    test_count = 128 if args.scale == 'full' else 64
    for i, row in enumerate(heldout):
        row['reference_selected'] = i < test_count
    math = local_rows(locations[DATASETS[2]], split='test')
    long_rows = []
    for index, row in enumerate(math):
        problem = row['problem']
        long_rows.append({'prompt_id': f'math500-long-{index:04d}', 'dataset': DATASETS[2],
                          'source_row': index, 'condition': 'long_thinking',
                          'text': 'Develop a detailed, self-contained mathematical chapter centered on the '
                                  'following problem. Derive the answer carefully. Give independent solution '
                                  'methods where valid, prove each supporting claim, work through boundary cases, '
                                  'then develop and solve ten distinct nontrivial variations. Include a complete '
                                  'Python verification program with property tests and explain its limitations. '
                                  'Aim for a substantial chapter with roughly 8,000 words of useful, nonrepetitive '
                                  'content. Avoid padding or repeating earlier passages.\n\nProblem:\n' + problem})
    long_rows = stable_order(long_rows, args.seed)
    long_dev_count, long_test_count = (32, 96) if args.scale == 'full' else (12, 32)
    for i, row in enumerate(long_rows):
        row['split'] = 'dev' if i < long_dev_count else 'test'
        row['reference_selected'] = i < long_dev_count + long_test_count
        row['drafter_distribution'] = 'out_of_distribution_thinking'
    result = {'scale': args.scale, 'seed': args.seed,
              'dataset_revisions': {d: env['dataset_artifacts'][d]['revision'] for d in DATASETS},
              'complete_heldout_dataset': DATASETS[1],
              'calibration_ids': calibration, 'calibration_condition': 'standard_nonthinking',
              'prompts': ordered + heldout + long_rows,
              'split_policy': f'GSM8K main train: deterministic 64 dev, remainder train; all HumanEval held-out test '
                              f'({test_count} selected); MATH-500 long: deterministic {long_dev_count} dev '
                              f'and {long_test_count} selected test, remainder test reserve.',
              'long_output_policy': 'natural EOS or 24576 cap; no min_new_tokens; detect/exclude repetition; '
                                    'report realized length and empty 8K/16K bins without filling them.'}
    ids = [row['prompt_id'] for row in result['prompts']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate prompt IDs')
    atomic_json(args.output / 'prompts.json', result)
    return result


def main():
    result = build_manifest(parse_config(common_parser(__doc__)))
    print(json.dumps({'scale': result['scale'], 'prompts': len(result['prompts']),
                      'calibration_prompts': len(result['calibration_ids'])}))


if __name__ == '__main__':
    main()

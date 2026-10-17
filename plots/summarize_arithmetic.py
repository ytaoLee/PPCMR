"""Derive the four-row Exact/FP32 difference table from local audit records."""
import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path


def summarize(dataset, path):
    audit = json.loads(path.read_text(encoding='utf-8'))
    assert audit['schema'] == 'ppcmr.plaintext_float_arithmetic_audit.v1'
    rows = []
    for direction, label in (('image-to-text', 'I2T'), ('text-to-image', 'T2I')):
        d = audit['directions'][direction]
        n = d['queries']
        equal = d['topk_order_equal_queries']
        assert n > 0 and all(0 <= equal[str(k)] <= n for k in (1, 5, 10))
        deltas = {
            k: (Decimal(str(d['exact_recall'][k])) - Decimal(str(d['fp32_recall'][k]))) * 100
            for k in ('R@1', 'R@5', 'R@10')
        }
        rows.append({'dataset': dataset, 'direction': label, 'queries': n,
                     'top10_order_differences': n - equal['10'],
                     'top1_id_differences': n - equal['1'],
                     **{f'delta_{k}_pp': str(v.normalize()) for k, v in deltas.items()}})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coco-audit', type=Path, required=True)
    p.add_argument('--flickr-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = summarize('COCO', a.coco_audit) + summarize('Flickr30k', a.flickr_audit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))

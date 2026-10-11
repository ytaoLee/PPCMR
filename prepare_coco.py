"""Create local COCO manifests directly from canonical Karpathy annotations."""
import argparse
import json
from pathlib import Path
import random

from research_common import ARTIFACTS, DATA_ROOT, file_sha256, value_sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--karpathy', type=Path)
    parser.add_argument('--output', type=Path, default=ARTIFACTS)
    parser.add_argument('--val-images', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.val_images < 1:
        parser.error('--val-images must be positive')
    root = args.data_root.resolve()
    source = args.karpathy or root / 'splits/karpathy/dataset_coco.json'
    document = json.loads(source.read_text(encoding='utf-8'))
    groups = {split: [row for row in document['images'] if row['split'] == split]
              for split in ('train', 'restval', 'val', 'test')}
    memberships = {split: {row['cocoid'] for row in rows} for split, rows in groups.items()}
    seen = set()
    for split, ids in memberships.items():
        if len(ids) != len(groups[split]) or seen & ids:
            raise ValueError('Duplicate or overlapping image IDs in Karpathy splits')
        seen |= ids
    if len(groups['test']) != 5000:
        raise ValueError('Expected the canonical 5000-image COCO test split')
    for split in ('val', 'test'):
        target = args.output / split / 'manifest.json'
        if target.exists():
            raise ValueError(f'Preserve the existing manifest: {target}')
    for split in ('val', 'test'):
        selected = groups[split]
        if split == 'val':
            indices = set(random.Random(args.seed).sample(range(len(selected)), min(args.val_images, len(selected))))
            selected = [row for i, row in enumerate(selected) if i in indices]
        shuffled = [row['cocoid'] for row in selected]
        random.Random(args.seed).shuffle(shuffled)
        scopes = {ident: f'scope-{i % 10:02d}' for i, ident in enumerate(shuffled)}
        images = []
        caption_ids = set()
        for row in selected:
            rel = f"raw/{row['filepath']}/{row['filename']}"
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Missing image or path outside data root: {path}')
            captions = [{'caption_id': c['sentid'], 'text': c['raw']} for c in row['sentences'][:5]]
            ids = [c['caption_id'] for c in captions]
            if len(captions) != 5 or len(set(ids)) != 5 or caption_ids & set(ids):
                raise ValueError('Each image must have five distinct original captions')
            if not all(isinstance(c['text'], str) and c['text'].strip() for c in captions):
                raise ValueError('Invalid caption text')
            caption_ids.update(ids)
            images.append({'image_id': row['cocoid'], 'image_path': rel,
                           'image_sha256': file_sha256(path), 'scope': scopes[row['cocoid']],
                           'captions': captions, 'source_caption_count': len(row['sentences'])})
        manifest = {'schema': 'ppcmr.local_coco.v1', 'data_root': str(root),
                    'dataset': 'COCO 2014 / original Karpathy split', 'split': split,
                    'seed': args.seed,
                    'selection': 'seeded validation subset' if split == 'val' else 'all 5000 test images in original Karpathy order',
                    'caption_rule': 'first five sentences in original dataset_coco.json order',
                    'permission_rule': '10 artificial balanced scopes independent of semantics; captions share image scope',
                    'image_count': len(images), 'text_count': 5 * len(images),
                    'images': images, 'source_files': {'karpathy': file_sha256(source)}}
        manifest['selection_sha256'] = value_sha256([(x['image_id'], [c['caption_id'] for c in x['captions']]) for x in images])
        write_json(args.output / split / 'manifest.json', manifest)
        print(f'Prepared {split}: {len(images)} images / {5 * len(images)} captions', flush=True)


if __name__ == '__main__':
    main()

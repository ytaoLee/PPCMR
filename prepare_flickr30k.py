"""Read local Flickr30k annotations and retain one declared test split."""
import argparse
import csv
import json
import random
from pathlib import Path

from research_common import file_sha256,value_sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/real_flickr30k/test')
    args=p.parse_args()
    root=args.data_root.resolve();out=args.output
    if (out/'manifest.json').exists():raise ValueError('Preserve existing manifest')
    source=root/'flickr_annotations_30k.csv'
    with source.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    subsets={s:{r['filename'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert all(not subsets[a]&subsets[b] for a,b in (('train','val'),('train','test'),('val','test')))
    chosen=[r for r in rows if r['split']=='test']
    assert len(chosen)==len(subsets['test'])==1000
    alternate={line.strip()+'.jpg' for line in (root/'test.txt').read_text().splitlines() if line.strip()}
    seed=20261010
    shuffled=[r['filename'] for r in chosen];random.Random(seed).shuffle(shuffled)
    scopes={name:f'scope-{i%10:02d}' for i,name in enumerate(shuffled)}
    images=[];sentids=set()
    for r in chosen:
        captions=json.loads(r['raw']);ids=json.loads(r['sentids'])
        assert len(captions)==len(ids)==5
        assert all(isinstance(c,str) and c.strip() for c in captions)
        assert not sentids&set(ids);sentids.update(ids)
        path=root/'flickr30k-images'/r['filename']
        assert path.is_file() and path.resolve().is_relative_to(root.resolve())
        images.append({'image_id':int(Path(r['filename']).stem),
            'annotation_image_id':int(r['img_id']),'image_path':str(path.relative_to(root)).replace('\\','/'),
            'image_sha256':file_sha256(path),'scope':scopes[r['filename']],
            'captions':[{'caption_id':int(i),'text':c} for i,c in zip(ids,captions)]})
    report={'schema':'ppcmr.local_flickr30k.v1','dataset':'Flickr30k',
        'data_root':str(root),'split':'provided annotation CSV test split','seed':seed,
        'selection':'all 1000 CSV test images in original CSV row order',
        'caption_rule':'all five raw captions and sentence IDs unchanged from annotation CSV',
        'permission_rule':'10 balanced artificial scopes independent of semantics',
        'image_count':1000,'text_count':5000,'images':images,
        'source_files':{'annotations':{'sha256':file_sha256(source)},
                        'alternate_test_list':{'sha256':file_sha256(root/'test.txt')}},
        'split_audit':{'CSV_counts':{k:len(v) for k,v in subsets.items()},
                       'CSV_split_overlap':0,'alternate_test_images':len(alternate),
                       'CSV_test_overlap_with_alternate':len(subsets['test']&alternate),
                       'rule':'CSV defines this experiment; the incompatible test.txt split is not merged.'}}
    report['selection_sha256']=value_sha256([(x['image_id'],[c['caption_id'] for c in x['captions']]) for x in images])
    write_json(out/'manifest.json',report)
    print('Flickr30k:',1000,'test images,',5000,'captions; alternate test list overlap',len(subsets['test']&alternate),flush=True)


if __name__=='__main__':main()

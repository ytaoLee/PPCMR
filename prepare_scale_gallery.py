"""Prepare a real, nested COCO scale gallery; reuse existing 5k embeddings."""
import argparse
import json
from pathlib import Path
import numpy as np
from research_common import file_sha256, value_sha256, write_json, load_features

BASE=Path(__file__).resolve().parent
OUT=BASE/'artifacts/scale_20261010'


def prepare():
    old=json.loads((BASE/'artifacts/real_coco/test/manifest.json').read_text(encoding='utf-8'))
    root=Path(old['data_root']).resolve()
    original=json.loads((root/'splits/karpathy/dataset_coco.json').read_text(encoding='utf-8'))
    train=[row for row in original['images'] if row['split']=='train']
    selected=np.random.default_rng(20261010).choice(len(train),15000,replace=False)
    extension=[]
    for i,index in enumerate(selected):
        source=train[int(index)]
        captions=[{'caption_id':c['sentid'],'text':c['raw']} for c in source['sentences'][:5]]
        rel=f"raw/{source['filepath']}/{source['filename']}"
        extension.append({'image_id':source['cocoid'],'image_path':rel,
            'image_sha256':file_sha256(root/rel),
            'scope':f'scope-{i%10:02d}','captions':captions,'source_split':'train'})
        if (i+1)%2500==0:print('Hashed extension images',i+1,flush=True)
    manifest={**{k:v for k,v in old.items() if k not in {'images','selection_sha256','source_files'}},
        'dataset':'COCO scalability extension','split':'extended-train-only','seed':20261010,
        'selection':'15000 train images selected without replacement; original IDs/order retained',
        'image_count':15000,'text_count':75000,'images':extension,
        'source_files':{'karpathy':file_sha256(root/'splits/karpathy/dataset_coco.json')},
        'selection_sha256':value_sha256([(r['image_id'],[c['caption_id'] for c in r['captions']]) for r in extension])}
    target=OUT/'extension/manifest.json'
    if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==manifest
    else:write_json(target,manifest)
    print('Extension manifest ready:',target,flush=True)


def combine():
    old,old_encoder,images,texts=load_features(BASE/'artifacts/real_coco/test')
    extension,encoder,extra_images,extra_texts=load_features(OUT/'extension')
    for field in ('weights_sha256','preprocess','tokenizer','dimensions','precision','normalization'):
        assert encoder[field]==old_encoder[field],field
    manifest={**old,'dataset':'COCO nested scalability gallery','split':'test+train-extension',
        'selection':'existing 5k Karpathy test in current order + 15k distinct train images',
        'image_count':20000,'text_count':100000,'images':old['images']+extension['images']}
    manifest['selection_sha256']=value_sha256([(r['image_id'],[c['caption_id'] for c in r['captions']]) for r in manifest['images']])
    assert len({r['image_id'] for r in manifest['images']})==20000
    folder=OUT/'gallery';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'manifest.json',manifest)
    merged_images=np.concatenate((images,extra_images));merged_texts=np.concatenate((texts,extra_texts))
    np.save(folder/'images.npy',merged_images);np.save(folder/'texts.npy',merged_texts)
    assert np.array_equal(merged_images[:5000],images) and np.array_equal(merged_texts[:25000],texts)
    write_json(folder/'encoder.json',{**encoder,'manifest_sha256':file_sha256(folder/'manifest.json'),
        'image_shape':list(merged_images.shape),'text_shape':list(merged_texts.shape),
        'features_sha256':{'images.npy':file_sha256(folder/'images.npy'),'texts.npy':file_sha256(folder/'texts.npy')},
        'existing_5k_features_reused_bitwise':True,'extension_encoder_sha256':file_sha256(OUT/'extension/encoder.json'),
        'retained_5k_encoder_sha256':file_sha256(BASE/'artifacts/real_coco/test/encoder.json')})
    print('Merged actual gallery:',merged_images.shape,merged_texts.shape,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--combine',action='store_true');a=p.parse_args()
    combine() if a.combine else prepare()

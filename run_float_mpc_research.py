"""Real COCO exact two-cloud sharing experiment; no model training required."""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import copy
import json
import platform
import importlib.metadata
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import exact_dot_encoded, encode_fp32
from ppcmr.crypto import canonical, decode, encode
from ppcmr.float_mpc import (TwoCloudEngine, expected_scores, make_query, publish, verify_result)
from research_common import (ARTIFACTS, DATA_ROOT, entity_arrays, file_sha256,
                             load_features, write_json)


def build_records(manifest, images, texts, count):
    root = Path(manifest['data_root']).resolve()
    records = []
    for i,item in enumerate(manifest['images'][:count]):
        image_path = (root/item['image_path']).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError('Image path escapes source dataset')
        if file_sha256(image_path) != item['image_sha256']:
            raise ValueError('Original image changed since feature extraction')
        records.append({'id':f"image-{item['image_id']:012d}",'scope':item['scope'],'modality':'image',
            'embedding':images[i], 'content':json.dumps({'type':'image/jpeg',
            'sha256':item['image_sha256'],'data':encode(image_path.read_bytes())},separators=(',',':'))})
        for j,caption in enumerate(item['captions']):
            records.append({'id':f"text-{caption['caption_id']:012d}",'scope':item['scope'],'modality':'text',
                'embedding':texts[5*i+j], 'content':json.dumps({'type':'text/plain','text':caption['text']},
                                                    ensure_ascii=False,separators=(',',':'))})
    return records


def tamper_cases(result, engine):
    variants = {}
    def add(name,change):
        bad = copy.deepcopy(result); change(bad); variants[name] = bad
    add('incorrect_score',lambda r:r['scores'].__setitem__(0,r['scores'][0]+1))
    add('omit_score',lambda r:r['scores'].pop())
    add('omit_authorized_candidate',lambda r:r['witnesses'].pop())
    add('duplicate_candidate',lambda r:r['witnesses'].append(copy.deepcopy(r['witnesses'][0])))
    add('alter_scope',lambda r:r['witnesses'][0].__setitem__('scope','scope-nonexistent'))
    if 'feature' in result['witnesses'][0]:
        add('alter_feature_ciphertext',lambda r:r['witnesses'][0]['feature'].__setitem__('ciphertext','AAAA'))
    else:
        add('alter_feature_digest',lambda r:r['witnesses'][0].__setitem__('feature_digest','0'*64))
    add('omit_topk_item',lambda r:r['results'].pop())
    add('reverse_topk_order',lambda r:r['results'].reverse())
    add('duplicate_topk_item',lambda r:r['results'].__setitem__(-1,copy.deepcopy(r['results'][0])))
    add('alter_content_ciphertext',lambda r:r['results'][0]['content'].__setitem__('ciphertext','AAAA'))
    add('wrong_snapshot',lambda r:r.__setitem__('snapshot_digest','0'*64))
    add('wrong_request',lambda r:r.__setitem__('request_digest','0'*64))
    target = result['witnesses'][0]['modality']
    extra = next((m for m in engine.server_data['records'] if m['modality']==target and
                  m['scope'] not in {w['scope'] for w in result['witnesses']}),None)
    if extra:
        add('inject_unauthorized_candidate',lambda r:r['witnesses'].append(copy.deepcopy(extra)))
        add('return_unauthorized_content',lambda r:r['results'].__setitem__(0,
                {'id':extra['id'],'content':engine.server_data['contents'][extra['id']]}))
    selected = {r['id'] for r in result['results']}
    replacement = next((m for m in result['witnesses'] if m['id'] not in selected),None)
    if replacement:
        add('replace_best_with_valid_lower_record',lambda r:r['results'].__setitem__(0,
            {'id':replacement['id'],'content':engine.server_data['contents'][replacement['id']]}))
    return variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ARTIFACTS/'test')
    p.add_argument('--images',type=int,nargs='+',default=[100,250])
    p.add_argument('--queries',type=int,default=3,help='Query pairs reused across directions, permissions and sizes')
    p.add_argument('--output',type=Path,default=ARTIFACTS/'float_mpc_v2')
    args = p.parse_args()
    manifest,provenance,images,texts = load_features(args.folder)
    ii,ti,il,tl,iscopes,tscopes = entity_arrays(manifest)
    all_rows, attacks, deployment = [],[],[]
    for n_images in args.images:
        if not 10 <= n_images <= len(images):
            raise ValueError('Image count outside cache bounds')
        records = build_records(manifest,images,texts,n_images)
        grants = {'none':[], 'one':['scope-00'],
            'half':[f'scope-{i:02d}' for i in range(5)],
            'all':[f'scope-{i:02d}' for i in range(10)]}
        started = perf_counter()
        server,shares,clients = publish(records,grants,{
            'model':provenance['model'],'features_sha256':provenance['features_sha256'],
            'manifest_sha256':provenance['manifest_sha256'],'subset_images':n_images})
        publish_seconds = perf_counter()-started
        with TwoCloudEngine(server,shares) as engine:
            deployment.append({'images':n_images,'records':len(records),'publish_seconds':publish_seconds,
                'worker_launch_seconds':engine.startup_seconds,
                'database_share_raw_bytes_both_clouds':len(records)*512*40*2})
            cohort = np.flatnonzero(iscopes[:n_images]=='scope-00')[:args.queries]
            if len(cohort) != args.queries:
                raise ValueError('Not enough common authorized query images')
            for subject in ('one','half','all'):
                client = clients[subject]
                for direction in ('image-to-text','text-to-image'):
                    for query_number,i in enumerate(cohort):
                        if direction == 'image-to-text':
                            feature,modality,label = images[i],'image',int(il[i])
                        else:
                            feature,modality,label = texts[5*i],'text',int(tl[5*i])
                        session = make_query(client,feature,modality,10)
                        # Plaintext independent reference from the ORIGINAL cache,
                        # not the cloud response or encrypted witness.
                        candidates = sorted([r for r in records if r['scope'] in grants[subject] and
                            r['modality'] != modality],key=lambda r:r['id'])
                        qenc = encode_fp32(feature)
                        start = perf_counter()
                        plain_scores = [exact_dot_encoded(qenc,encode_fp32(r['embedding'])) for r in candidates]
                        order = sorted(range(len(candidates)),key=lambda j:(-plain_scores[j],candidates[j]['id']))[:10]
                        plain_ids = [candidates[j]['id'] for j in order]
                        plain_seconds = perf_counter()-start
                        fp32_scores = np.stack([r['embedding'] for r in candidates]) @ feature
                        fp32_order = np.argsort(-fp32_scores,kind='stable')[:10]
                        fp32_ids = [candidates[j]['id'] for j in fp32_order]
                        response,cost = engine.search(client,session)
                        start = perf_counter()
                        verified = verify_result(client,session,response)
                        verification_seconds = perf_counter()-start
                        if not verified['accepted']:
                            raise RuntimeError(verified)
                        exact_equal = response['scores'] == plain_scores
                        order_equal = verified['ids'] == plain_ids
                        if not exact_equal or not order_equal:
                            raise RuntimeError('MPC differs from independent exact plaintext reference')
                        # The returned content includes original JPEG bytes or captions.
                        for content in verified['contents']:
                            body = json.loads(content)
                            if body['type']=='image/jpeg':
                                import hashlib
                                if hashlib.sha256(decode(body['data'])).hexdigest()!=body['sha256']:
                                    raise RuntimeError('Returned JPEG bytes do not match the original')
                        row = {'images':n_images,'permission':subject,'direction':direction,
                            'query_number':query_number,'query_image_label':label,
                            'exact_scores_equal':exact_equal,'exact_topk_order_equal':order_equal,
                            'fp32_topk_order_equal':verified['ids']==fp32_ids,'verification_accepted':True,
                            'verification_seconds':verification_seconds,'plain_exact_seconds':plain_seconds,**cost}
                        all_rows.append(row)
                        if n_images==args.images[0] and subject=='one' and query_number==0:
                            for name,bad in tamper_cases(response,engine).items():
                                verdict = verify_result(client,session,bad)
                                attacks.append({'direction':direction,'attack':name,
                                    'rejected':not verdict['accepted'],'reason':verdict['reason']})
                            try:
                                engine.search(client,session)
                            except ValueError as error:
                                attacks.append({'direction':direction,'attack':'replayed_query',
                                                'rejected':True,'reason':str(error)})
                            else:
                                attacks.append({'direction':direction,'attack':'replayed_query','rejected':False})
                            write_json(args.output/f'sample_{direction}_request.json',session.request)
                            write_json(args.output/f'sample_{direction}_response.json',response)
                            write_json(args.output/f'sample_{direction}_verdict.json',
                                       {k:v for k,v in verified.items() if k!='contents'})
                    print(n_images,subject,direction,'passed',len(cohort),'queries',flush=True)
            for modality in ('image','text'):
                feature = images[cohort[0]] if modality=='image' else texts[5*cohort[0]]
                session = make_query(clients['none'],feature,modality,10)
                response,cost = engine.search(clients['none'],session)
                verified = verify_result(clients['none'],session,response)
                if not verified['accepted'] or response['results'] or response['scores'] or response['witnesses']:
                    raise RuntimeError('Zero-rights query did not produce an authenticated empty result')
                all_rows.append({'images':n_images,'permission':'none','direction':modality+'-query',
                    'query_number':0,'exact_scores_equal':True,'exact_topk_order_equal':True,
                    'fp32_topk_order_equal':True,'verification_accepted':True,**cost})
    if not all(a['rejected'] for a in attacks):
        raise RuntimeError('A configured attack was accepted')
    honest = [r for r in all_rows if r['permission']!='none']
    report = {'schema':'ppcmr.two-cloud-exact-float.experiment.v2',
        'query_binding':'shared signed v2 context; recipient-private signed query-share envelopes',
        'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                   'numpy':np.__version__,'cryptography':importlib.metadata.version('cryptography')},
        'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_sha256(p) for p in
            (Path(__file__),Path(__file__).parent/'ppcmr/float_mpc.py',Path(__file__).parent/'ppcmr/crypto.py')},
        'dataset':'COCO Karpathy test-cache subsets; artificial scopes',
        'model':provenance['model'],'features_sha256':provenance['features_sha256'],
        'manifest_sha256':provenance['manifest_sha256'],'training_performed':False,
        'dimensions':512,'arithmetic':'lossless FP32-value encoding x*2^149; exact inner product in Z/(2^320)',
        'privacy_model':'two non-colluding semi-honest servers; trusted fresh matrix-vector triple dealer',
        'simulation':'two distinct Windows worker processes on one physical host, local private IPC',
        'topk_location':'client, reconstructing all authorized scores',
        'verification':'full owner-signed authorized encrypted-feature witnesses, O(N) client recomputation',
        'not_implemented':['secure top-k sorting inside the cloud parties','succinct verification',
            'malicious-secure MPC privacy proof','dealer-free preprocessing','remote TLS deployment'],
        'honest_nonempty_queries':len(honest),'authenticated_empty_queries':len(all_rows)-len(honest),
        'exact_score_equal_queries':sum(r['exact_scores_equal'] for r in honest),
        'exact_topk_order_equal_queries':sum(r['exact_topk_order_equal'] for r in honest),
        'fp32_topk_order_equal_queries':sum(r['fp32_topk_order_equal'] for r in honest),
        'configured_attacks':len(attacks),'attacks_rejected':sum(a['rejected'] for a in attacks),
        'deployment':deployment,'measurements':all_rows,'attacks':attacks}
    write_json(args.output/'experiment.json',report)
    print({k:v for k,v in report.items() if k not in ('measurements','attacks','deployment')},flush=True)


if __name__ == '__main__':
    main()

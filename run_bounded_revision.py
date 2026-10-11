"""One bounded final-profile cohort and one real epoch lifecycle.

Three distinct fixed queries per direction, one measured execution each.
No parameter sweep or repeated attack matrix. All observations are archived.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'

import argparse
import gc
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32,exact_dot_encoded
from ppcmr.cached_local import CachedLocalEngine
from ppcmr.cached_mpc import CachedMatrixEngine
from ppcmr.compact_verify import prepare_verifier_state
from ppcmr.crypto import (canonical,decrypt_content,encode,new_owner_key)
from ppcmr.float_mpc import TwoCloudEngine,aad,make_query,publish
from ppcmr.publication_policy import POLICY
from ppcmr.windows_memory import MemorySampler
from research_common import file_sha256,load_features,write_json
from run_cached_local_research import perform_cached_local
from run_compact_research import perform

BASE=Path(__file__).resolve().parent
OUT=BASE/'artifacts/final_revision_20261010/final'
METHODS=('fresh_mpc','cached_mpc','cached_local_limb')


def records_for(manifest,images,texts,size):
    root=Path(manifest['data_root']).resolve();records=[]
    for i,item in enumerate(manifest['images'][:size]):
        path=(root/item['image_path']).resolve()
        if not path.is_relative_to(root) or file_sha256(path)!=item['image_sha256']:
            raise ValueError('Source image changed')
        records.append({'id':f"image-{item['image_id']:012d}",'scope':item['scope'],'modality':'image',
            'embedding':images[i],'content':json.dumps({'type':'image/jpeg','sha256':item['image_sha256'],
                                                       'data':encode(path.read_bytes())},separators=(',',':'))})
        for j,c in enumerate(item['captions']):
            records.append({'id':f"text-{c['caption_id']:012d}",'scope':item['scope'],'modality':'text',
                'embedding':texts[5*i+j],
                'content':json.dumps({'type':'text/plain','text':c['text']},ensure_ascii=False,separators=(',',':'))})
    return records


def references(records,images,texts,indices,allowed):
    result={}
    for modality in ('image','text'):
        candidates=[r for r in sorted(records,key=lambda r:r['id'])
                    if r['modality']!=modality and r['scope'] in allowed]
        encoded=[encode_fp32(r['embedding']) for r in candidates]
        for i0 in indices:
            i=int(i0);feature=images[i] if modality=='image' else texts[5*i]
            q=encode_fp32(feature);scores=[exact_dot_encoded(q,x) for x in encoded]
            order=sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
            result[(modality,i)]={'scores':scores,'ids':[candidates[j]['id'] for j in order],
                                 'contents':[candidates[j]['content'] for j in order]}
    return result


def cohort(dataset,folder,size,index,report,save):
    manifest,provenance,images,texts=load_features(folder)
    expected=json.loads((BASE/'artifacts/real_coco/test/encoder.json').read_text())
    for field in ('weights_sha256','preprocess','tokenizer','dimensions','precision'):
        if provenance[field]!=expected[field]:raise ValueError('Encoder differs across datasets: '+field)
    records=records_for(manifest,images,texts,size)
    indices=[int(i) for i in np.random.default_rng(20261010).permutation(size)[:3]]
    rights=[f'scope-{i:02d}' for i in range(10)]
    refs=references(records,images,texts,indices,rights)
    start=perf_counter()
    server,shares,clients=publish(records,{'all':rights},provenance,compact=True)
    publication=perf_counter()-start;client=clients['all']
    detail={'dataset':dataset,'images':size,'captions':5*size,'query_indices':indices,
            'publication_seconds':publication,'publication_policy':POLICY,
            'private_state_JSON_bytes':len(canonical({f'{a}:{m}':v for (a,m),v in client.verification_sketches.items()})),
            'features_sha256':provenance['features_sha256'],'manifest_sha256':provenance['manifest_sha256'],
            'setup':{},'measurements':[],'warmups':[],'complete':False}
    report['cohorts'].append(detail);save()
    order=METHODS[index%3:]+METHODS[:index%3]
    with MemorySampler() as memory:
        for method in order:
            memory.stage=method+'_setup'
            start=perf_counter();state=prepare_verifier_state(client) if method!='cached_local_limb' else None
            init_state=perf_counter()-start
            Engine={'fresh_mpc':TwoCloudEngine,'cached_mpc':CachedMatrixEngine,'cached_local_limb':CachedLocalEngine}[method]
            instance=Engine(server) if method=='cached_local_limb' else Engine(server,shares)
            with instance as engine:
                memory.register_workers(engine,method)
                setup={'private_state_initialization_seconds':init_state,'worker_launch_seconds':engine.startup_seconds}
                if method=='cached_mpc':setup['matrix']=engine.setup_metrics
                caches={}
                if method=='cached_local_limb':
                    setup['local_cache']={}
                    for modality in ('image','text'):
                        caches[modality],setup['local_cache'][modality]=engine.initialize(client,modality)
                detail['setup'][method]=setup
                memory.stage=method+'_query'
                def run(modality,i):
                    feature=images[i] if modality=='image' else texts[5*i]
                    if method=='cached_local_limb':
                        return perform_cached_local(engine,client,caches[modality],feature,modality,refs[(modality,i)],method)
                    return {'method':method,**perform(engine,client,state,feature,modality,refs[(modality,i)],'compact')[2]}
                for modality in ('image','text'):
                    detail['warmups'].append({'method':method,'query_modality':modality,
                        'query_cache_index':indices[0],**run(modality,indices[0])})
                    for number,i in enumerate(indices):
                        row=run(modality,i)
                        detail['measurements'].append({'query_modality':modality,'query_number':number,
                            'query_cache_index':i,'method_block_order':order,**row})
                        save()
                    print(dataset,size,method,modality,'three queries exact',flush=True)
                memory.sample()
            del engine,instance,state,caches
            gc.collect();memory.sample()
        detail['memory']=memory.report()
    detail['complete']=True;save()
    del refs,records,server,shares,clients,client
    gc.collect()


def epoch_lifecycle(report,save):
    manifest,provenance,images,texts=load_features(BASE/'artifacts/real_coco/test')
    records=records_for(manifest,images,texts,100);owner=new_owner_key()
    rights=[f'scope-{i:02d}' for i in range(10)]
    out={'images':100,'epochs':[],'queries':[],'complete':False};report['epoch_lifecycle']=out
    old_client=None;old_public=None
    for epoch,allowed in ((1,rights),(2,rights[:5])):
        start=perf_counter()
        server,shares,clients=publish(records,{'u':allowed},provenance,epoch=epoch,owner_key=owner,compact=True)
        publication=perf_counter()-start;client=clients['u']
        refs=references(records,images,texts,[8],allowed)
        start=perf_counter();state=prepare_verifier_state(client);state_seconds=perf_counter()-start
        with CachedMatrixEngine(server,shares) as engine:
            out['epochs'].append({'epoch':epoch,'allowed_scopes':len(allowed),'publication_seconds':publication,
                'private_state_initialization_seconds':state_seconds,'matrix_setup':engine.setup_metrics,
                'worker_PIDs':[p.pid for p in engine.processes]})
            if epoch==2:
                assert client.owner_public==old_public
                try:engine.search(old_client,make_query(old_client,images[8],'image'),'compact')
                except ValueError as error:
                    assert 'Stale' in str(error);out['old_credential_rejected']=True
                else:raise AssertionError('Old credential accepted')
                m=next(m for m in server['records'] if m['scope']=='scope-05')
                try:decrypt_content(old_client.scope_keys['scope-05'],m['feature'],aad(server['snapshot']['payload'],m,'feature'))
                except Exception:out['old_scope_key_cannot_decrypt_new_ciphertext']=True
                else:raise AssertionError('Old key decrypts new epoch')
            for modality in ('image','text'):
                feature=images[8] if modality=='image' else texts[40]
                row=perform(engine,client,state,feature,modality,refs[(modality,8)],'compact')[2]
                out['queries'].append({'epoch':epoch,'query_modality':modality,**row})
        if epoch==1:old_client=client;old_public=client.owner_public
        save()
    out['complete']=True;save();print('epoch lifecycle: four exact queries, old credential/key rejected',flush=True)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output
    destination=OUT/'experiment.json'
    if destination.exists():raise ValueError('Keep existing executed evidence; select a new folder')
    source_names=['run_bounded_revision.py','ppcmr/windows_memory.py','run_cached_local_research.py',
                  'run_compact_research.py','research_common.py','float_reference.py','ppcmr/crypto.py',
                  'ppcmr/publication_policy.py','ppcmr/float_mpc.py','ppcmr/compact_verify.py',
                  'ppcmr/cached_mpc.py','ppcmr/exact_ring.py','ppcmr/cached_local.py','ppcmr/local_reference.py']
    report={'schema':'ppcmr.bounded-final-profile.v1','date':'2026-10-10','publication_policy':POLICY,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'BLAS_threads':4},
        'cohorts':[],'complete':False,
        'source_sha256':{name:file_sha256(BASE/name) for name in source_names},
        'design':'Three fixed distinct query pairs per cohort, one measured execution each and one excluded warm-up per direction/method. Same publication within cohort; sequential method blocks, rotated by cohort. Complete exact scores/order/content checked independently.',
        'scope':'Same-host IPC; Windows working set measurements separate workers from combined trusted harness. No isolated client/dealer memory or WAN experiment.',
        'budget':{'measured_comparisons':72,'excluded_warmups':24,'epoch_normal_queries':4}}
    report['partial_attempts']=[
        {'file':'../experiment.json','measured_executions':12,'warmups':4,
         'reason':'Driver passed shares as a local-kernel argument.'},
        {'file':'../completed/experiment.json','measured_executions':12,'warmups':4,
         'reason':'Memory adapter expected a process list rather than the local engine single process.'}]
    report['partial_attempt_rule']='Incomplete pilot reports retained; no rows substituted for this complete cohort.'
    def save():write_json(destination,report)
    save()
    for index,(dataset,folder,size) in enumerate((
            ('COCO',BASE/'artifacts/real_coco/test',100),
            ('COCO',BASE/'artifacts/real_coco/test',250),
            ('COCO',BASE/'artifacts/real_coco/test',1000),
            ('Flickr30k',BASE/'artifacts/real_flickr30k/test',1000))):
        print('begin',dataset,size,flush=True)
        cohort(dataset,folder,size,index,report,save)
    epoch_lifecycle(report,save)
    assert sum(len(c['measurements']) for c in report['cohorts'])==72
    report['complete']=True;save();print('bounded experiment complete',flush=True)


if __name__=='__main__':main()

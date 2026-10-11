"""Four measured real COCO gallery sizes, bounded memory, unchanged exactness."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from research_common import file_sha256,write_json
from float_reference import encode_fp32,exact_dot_encoded
from ppcmr.crypto import canonical,encode
from ppcmr.float_mpc import make_query
from ppcmr.compact_verify import prepare_verifier_state,verify_compact
from ppcmr.streaming_store import publish_streamed
from ppcmr.streaming_mpc import StreamingEngine
from ppcmr.streaming_local import StreamingLocalEngine
from ppcmr.local_reference import make_local_query
from ppcmr.windows_memory import MemorySampler

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
GALLERY=BASE/'artifacts/scale_20261010/gallery'
METHODS=('fresh_mpc','cached_mpc','cached_local_limb')
QUERIES=(989,32,715)


class ScaleRecords:
    def __init__(self,manifest,images,texts,size):
        self.manifest=manifest;self.images=images;self.texts=texts;self.root=Path(manifest['data_root'])
        self.descriptors=[]
        for i,item in enumerate(manifest['images'][:size]):
            self.descriptors.append({'id':f"image-{item['image_id']:012d}",'scope':item['scope'],'modality':'image','image_index':i})
            for j,c in enumerate(item['captions']):
                self.descriptors.append({'id':f"text-{c['caption_id']:012d}",'scope':item['scope'],'modality':'text','image_index':i,'caption_index':j})
        self.descriptors.sort(key=lambda r:r['id']);self.lookup={r['id']:i for i,r in enumerate(self.descriptors)}
    def __len__(self):return len(self.descriptors)
    def feature(self,r):
        i=r['image_index'];return self.images[i] if r['modality']=='image' else self.texts[5*i+r['caption_index']]
    def __getitem__(self,index):
        r=self.descriptors[index];item=self.manifest['images'][r['image_index']]
        if r['modality']=='image':
            path=(self.root/item['image_path']).resolve()
            if not path.is_relative_to(self.root.resolve()):raise ValueError('Image path outside dataset')
            data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==item['image_sha256']
            content=json.dumps({'type':'image/jpeg','sha256':item['image_sha256'],'data':encode(data)},separators=(',',':'))
        else:content=json.dumps({'type':'text/plain','text':item['captions'][r['caption_index']]['text']},ensure_ascii=False,separators=(',',':'))
        return {**{k:r[k] for k in ('id','scope','modality')},'embedding':self.feature(r),'content':content}


def reference(records,images,texts):
    refs={}
    for modality in ('image','text'):
        candidates=[r for r in records.descriptors if r['modality']!=modality]
        queries=[encode_fp32(images[i] if modality=='image' else texts[5*i]) for i in QUERIES]
        scores=[[] for _ in QUERIES]
        for r in candidates:
            encoded=encode_fp32(records.feature(r))
            for target,q in zip(scores,queries):target.append(exact_dot_encoded(q,encoded))
        for i,values in zip(QUERIES,scores):
            order=sorted(range(len(candidates)),key=lambda j:(-values[j],candidates[j]['id']))[:10]
            ids=[candidates[j]['id'] for j in order]
            refs[(modality,i)]={'scores':values,'ids':ids,'contents':[records[records.lookup[ident]]['content'] for ident in ids]}
    return refs


def checked_query(engine,client,state,cache,feature,modality,expected,method):
    t=perf_counter()
    if method=='cached_local_limb':
        session=make_local_query(client,feature,modality);construction=perf_counter()-t
        result,cost=engine.search_cached(client,session,cache);complete=perf_counter()-t
        ids=result['ids'];scores=result['scores'];contents=result['contents'];verification=0.
    else:
        session=make_query(client,feature,modality);construction=perf_counter()-t
        result,cost=engine.search(client,session);verified=verify_compact(client,session,result,state)
        complete=perf_counter()-t;assert verified['accepted'],verified['reason']
        ids=verified['ids'];scores=result['scores'];contents=verified['contents'];verification=verified['seconds']
    assert scores==expected['scores'],'Any exact score differs'
    assert ids==expected['ids'],'Exact deterministic top-k differs'
    assert contents==expected['contents'],'Decrypted original content differs'
    return {'method':method,'query_construction_seconds':construction,'online_total_seconds':complete,
        'verification_seconds':verification,**cost,'exact_scores_equal':True,'exact_order_equal':True,'original_contents_equal':True}


def cleanup_created(folder):
    folder=Path(folder).resolve();allowed=(BASE/'artifacts/scale_20261010').resolve()
    assert folder.is_relative_to(allowed) and folder.name.startswith('scratch_')
    # Only this runner's new scratch files; no recursive shell deletion.
    for path in folder.rglob('*'):
        if path.is_file():
            assert path.resolve().is_relative_to(folder);path.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--sizes',type=int,nargs='+',default=[1000,5000,10000,20000])
    p.add_argument('--block-rows',type=int,default=256)
    p.add_argument('--output',type=Path,default=BASE/'artifacts/scale_20261010/executed')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report_path=a.output/'experiment.json'
    if report_path.exists():raise ValueError('Keep existing evidence; use a new output folder')
    manifest=json.loads((GALLERY/'manifest.json').read_text(encoding='utf-8'))
    provenance=json.loads((GALLERY/'encoder.json').read_text(encoding='utf-8'))
    assert file_sha256(GALLERY/'manifest.json')==provenance['manifest_sha256']
    for name,expected in provenance['features_sha256'].items():assert file_sha256(GALLERY/name)==expected
    images=np.load(GALLERY/'images.npy',mmap_mode='r');texts=np.load(GALLERY/'texts.npy',mmap_mode='r')
    sources=['run_scale_research.py','prepare_scale_gallery.py','ppcmr/streaming_store.py','ppcmr/streaming_mpc.py',
        'ppcmr/streaming_local.py','ppcmr/exact_ring.py','ppcmr/cached_local.py','ppcmr/float_mpc.py',
        'ppcmr/compact_verify.py','ppcmr/crypto.py','float_reference.py','ppcmr/windows_memory.py']
    report={'schema':'ppcmr.streamed-scale.v1','date':'2026-10-10','complete':False,'cohorts':[],
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'BLAS_threads':4},
        'source_sha256':{name:file_sha256(BASE/name) for name in sources},'encoder':provenance,
        'design':'Same three query pairs at all nested sizes; three methods, two directions, one measurement each and one excluded warm-up each. Sequential method blocks rotated by size. All scores independently checked with Python integers; complete contents and ordered IDs checked.',
        'scope':'Same-host private IPC and role-separated disk stores, CPU exact scoring; fixed-gallery CLIP extraction excluded. Disk IO is included in publication/setup/query intervals. Bootstrap file mapping/copying and common publication excluded from query costs.',
        'timing':'Complete-query wall time includes interleaved trusted preprocessing, IPC, authentication, all exact scores, local ranking and content verification. Component times may overlap the complete interval and are not added.',
        'gallery_definition':'1k/5k are current Karpathy-test prefixes; 10k/20k append distinct training images for scalability only. No additional training, quantization, duplicate records, or approximate pruning.',
        'block_rows':a.block_rows,'queries':list(QUERIES),'configured_sizes':a.sizes}
    def save():write_json(report_path,report)
    save()
    for cohort_index,size in enumerate(a.sizes):
        assert size in (1000,5000,10000,20000)
        print('Begin true gallery',size,'images',5*size,'captions',flush=True)
        records=ScaleRecords(manifest,images,texts,size)
        t=perf_counter();refs=reference(records,images,texts);reference_seconds=perf_counter()-t
        print('Independent exact references ready',size,round(reference_seconds,2),'s',flush=True)
        scratch=BASE/'artifacts/scale_20261010'/f'scratch_{a.output.name}_{size}'
        with MemorySampler() as memory:
            memory.stage='publication'
            server,shares,clients,publication=publish_streamed(records,{'all':[f'scope-{i:02d}' for i in range(10)]},scratch,a.block_rows)
            client=clients['all'];detail={'dataset':'COCO','images':size,'captions':size*5,
                'gallery_kind':'Karpathy-test prefix' if size<=5000 else 'extended scalability gallery',
                'query_indices':list(QUERIES),'independent_reference_seconds':reference_seconds,'publication':publication,
                'private_state_JSON_bytes':len(canonical({f'{s}:{m}':v for (s,m),v in client.verification_sketches.items()})),
                'measurements':[],'warmups':[],'setup':{},'complete':False}
            report['cohorts'].append(detail);save()
            order=METHODS[cohort_index%3:]+METHODS[:cohort_index%3]
            for method in order:
                memory.stage=method+'_setup';t=perf_counter()
                state=prepare_verifier_state(client) if method!='cached_local_limb' else None
                private_init=perf_counter()-t;t=perf_counter();folder=scratch/method
                instance=(StreamingLocalEngine(server,folder,a.block_rows) if method=='cached_local_limb' else
                    StreamingEngine(server,shares,folder,'fresh' if method=='fresh_mpc' else 'cached',a.block_rows))
                caches={}
                with instance as engine:
                    memory.register_workers(engine,method)
                    setup={'private_state_initialization_seconds':private_init,'worker_launch_seconds':engine.startup_seconds}
                    if method=='cached_mpc':setup['matrix']=engine.setup_metrics
                    if method=='cached_local_limb':
                        setup['local_cache']={}
                        for modality in ('image','text'):caches[modality],setup['local_cache'][modality]=engine.initialize(client,modality)
                    setup['two_modality_setup_seconds']=perf_counter()-t+private_init;detail['setup'][method]=setup;save()
                    for modality in ('image','text'):
                        memory.stage=method+'_'+modality
                        for query_number,i in enumerate((QUERIES[0],*QUERIES)):
                            feature=images[i] if modality=='image' else texts[5*i]
                            row=checked_query(engine,client,state,caches.get(modality),feature,modality,refs[(modality,i)],method)
                            row.update({'query_modality':modality,'query_index':i,'method_order':order})
                            if query_number==0:detail['warmups'].append(row)
                            else:row['query_number']=query_number-1;detail['measurements'].append(row)
                            save()
                        print('Exact measured block',size,method,modality,flush=True)
                    memory.sample()
                del engine,instance,caches,state;gc.collect();memory.sample()
            detail['memory']=memory.report();detail['complete']=True;save()
        server['contents'].close();server['metadata_store'].close()
        del server,shares,clients,client,refs,records;gc.collect();cleanup_created(scratch)
    assert sum(len(c['measurements']) for c in report['cohorts'])==len(a.sizes)*18
    report['complete']=True;save();print('All configured scale cohorts completed exactly',flush=True)


if __name__=='__main__':main()

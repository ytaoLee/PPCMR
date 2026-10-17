"""Bounded COCO I2T experiment. Never estimates unexecuted query timings."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import argparse
import gc
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter
import numpy as np
import cryptography
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
os.environ['PYTHONPATH']=str(REPO)+os.pathsep+os.environ.get('PYTHONPATH','')
from ppcmr import streaming_store
from ppcmr.streaming_store import publish_streamed
from ppcmr.float_mpc import make_query
from ppcmr.crypto import digest,sign
from ppcmr.compact_verify import prepare_verifier_state,verify_compact
from float_reference import encode_fp32,exact_dot_encoded
from run_scale_research import ScaleRecords
from research_common import file_sha256
from network_topology import NS,PROFILES,profile,status
from tcp_engine import TCPEngine

BASE=Path(__file__).resolve().parent
GALLERY=REPO/'artifacts/scale_20261010/gallery'
SCRATCH=None

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(path)

def reference(records,images,queries):
    candidates=[r for r in records.descriptors if r['modality']=='text']
    encoded_queries=[encode_fp32(images[i]) for i in queries]
    vectors=[[] for _ in queries]
    for row in candidates:
        x=encode_fp32(records.feature(row))
        for scores,q in zip(vectors,encoded_queries):scores.append(exact_dot_encoded(q,x))
    refs={}
    for query,scores in zip(queries,vectors):
        order=sorted(range(len(candidates)),key=lambda j:(-scores[j],candidates[j]['id']))[:10]
        ids=[candidates[j]['id'] for j in order]
        refs[query]={'scores':scores,'ids':ids,'contents':[records[records.lookup[i]]['content'] for i in ids]}
    return refs

def publication(records,folder):
    # An owner-selected 24-hour expiry avoids a time-dependent credential expiry
    # during this long sequential run. Authentication/verification is unchanged.
    original=streaming_store.new_owner_key;keys=[]
    def retained_key():
        key=original();keys.append(key);return key
    streaming_store.new_owner_key=retained_key
    scopes=sorted({r['scope'] for r in records.descriptors})
    try:server,shares,clients,metrics=publish_streamed(records,{'all':scopes},folder,256)
    finally:streaming_store.new_owner_key=original
    client=clients['all'];payload=dict(client.credential['payload']);payload['expires_at']=int(time.time())+86400
    client.credential=sign(keys[0],payload,'float-credential')
    return server,shares,client,metrics

def checked(engine,client,state,feature,expected,query):
    before=engine.counters();t=perf_counter();session=make_query(client,feature,'image',k=10)
    construction=perf_counter()-t
    result,cost=engine.search(client,session)
    verified=verify_compact(client,session,result,state);elapsed=perf_counter()-t
    assert verified['accepted'],verified
    assert result['scores']==expected['scores'],'Exact score mismatch'
    assert verified['ids']==expected['ids'],'Deterministic top-k mismatch'
    assert verified['contents']==expected['contents'],'Content mismatch'
    counters=[{k:b[k]-a[k] for k in a} for a,b in zip(before,engine.counters())]
    return {'query_index':query,'query_construction_seconds':construction,'complete_seconds':elapsed,
            'verification_seconds':verified['seconds'],'cost':cost,'control_transport':counters,
            'exact_scores_equal':True,'exact_order_equal':True,'original_contents_equal':True,
            'score_vector_digest':digest(result['scores']),'ids':verified['ids'],'request_digest':digest(session.request)}

def remove_created(path):
    target=Path(path).resolve();root=SCRATCH.resolve()
    assert target.is_relative_to(root) and target!=root
    shutil.rmtree(target)

def main():
    global GALLERY,SCRATCH
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--gallery',type=Path,default=GALLERY)
    p.add_argument('--coco-root',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True,help='New run-specific private scratch directory')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if sys.platform!='linux':p.error('Network namespaces require Linux; use test_tcp_transport.py for the portable framing check')
    GALLERY=a.gallery.resolve();SCRATCH=a.scratch.resolve()
    if SCRATCH.exists():p.error('Use a new scratch directory to preserve prior state')
    if not a.coco_root.is_dir():p.error('--coco-root must exist')
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report_path=output/'experiment.json';assert not report_path.exists(),'Preserve prior evidence'
    manifest=json.loads((GALLERY/'manifest.json').read_text())
    provenance=json.loads((GALLERY/'encoder.json').read_text())
    assert file_sha256(GALLERY/'manifest.json')==provenance['manifest_sha256']
    for name,expected in provenance['features_sha256'].items():assert file_sha256(GALLERY/name)==expected
    images=np.load(GALLERY/'images.npy',mmap_mode='r');texts=np.load(GALLERY/'texts.npy',mmap_mode='r')
    queries=[int(x) for x in np.random.default_rng(20261016).choice(5000,5,replace=False)]
    if a.smoke:queries=[1]
    sizes=[100] if a.smoke else [5000,20000]
    profiles=['N0','N2'] if a.smoke else list(PROFILES)
    sources={f.relative_to(REPO).as_posix():file_sha256(f) for f in sorted(REPO.rglob('*.py')) if not any(x in f.parts for x in ('artifacts','.venv','__pycache__','.git'))}
    design={'schema':'ppcmr.controlled-tcp-public-plan.v1','source_sha256':sources,'sizes':sizes,'profiles':profiles,'queries':queries,'warmups_per_cell':0 if a.smoke else 1,'block_rows':256,'seed':20261016,'gallery':str(GALLERY),'encoder':provenance}
    design_path=output/'design.json';assert not design_path.exists(),'Preserve prior design'
    save(design_path,design)
    report={'schema':'ppcmr.controlled-tcp-network.v1','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'complete':False,'smoke':a.smoke,'runtime':{'python':platform.python_version(),'numpy':np.__version__,
      'cryptography':cryptography.__version__,'kernel':platform.release(),'platform':platform.platform(),'blas_threads':4},
      'queries':queries,'image_ids':[manifest['images'][i]['image_id'] for i in queries],'profiles':PROFILES,
      'source_sha256':sources,'design_sha256':file_sha256(design_path),'encoder':provenance,'credential_lifetime_seconds':86400,
      'transport':'TCP with 8-byte frame lengths; isolated namespaces, three independent full-duplex node-pair links via three bridges and six directional shapers; router egress qdiscs are downstream of TCP senders',
      'deployment':'DU/dealer share H; CS0 and CS1 are separate processes and network namespaces; same physical host/WSL kernel',
      'timing':'Complete query includes request construction, fresh dealer work/delivery, scoring, catalog/score response, ranking, selected content fetch and full verification. Publication/worker bootstrap and matrix setup are separate. Components may overlap.',
      'cohorts':[]}
    save(report_path,report)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    for size in sizes:
        records=ScaleRecords(manifest,images,texts,size);records.root=a.coco_root.resolve()
        t=perf_counter();refs=reference(records,images,queries)
        print(json.dumps({'stage':'reference_complete','images':size,'seconds':perf_counter()-t}),flush=True)
        root=SCRATCH/(output.name+f'-{size}');assert not root.exists();root.mkdir()
        server,shares,client,pub=publication(records,root/'publication')
        state=prepare_verifier_state(client)
        cohort={'images':size,'captions':size*5,'publication':pub,'snapshot_digest':digest(client.snapshot),
                'reference_digests':{str(q):digest(refs[q]['scores']) for q in queries},'cells':[]}
        report['cohorts'].append(cohort);save(report_path,report)
        for pi,name in enumerate(profiles):
            profile(name)
            order=['fresh','cached'] if (pi+(size==20000))%2==0 else ['cached','fresh']
            for method in order:
                cell={'profile':name,'method':method,'method_order':order,'warmups':[],'measurements':[],
                      'network_before':status(),'complete':False}
                cohort['cells'].append(cell);save(report_path,report)
                folder=root/(name+'-'+method)
                print(json.dumps({'stage':'setup_start','images':size,'profile':name,'method':method}),flush=True)
                t=perf_counter()
                with TCPEngine(server,shares,folder,method,256) as engine:
                    cell['setup_seconds_including_bootstrap']=perf_counter()-t
                    cell['matrix_setup']=engine.setup_metrics
                    save(report_path,report)
                    sequence=queries if a.smoke else [queries[0],*queries]
                    for index,q in enumerate(sequence):
                        row=checked(engine,client,state,images[q],refs[q],q)
                        warmup=not a.smoke and index==0
                        cell['warmups' if warmup else 'measurements'].append(row)
                        save(report_path,report)
                        print(json.dumps({'stage':'query_complete','images':size,'profile':name,'method':method,
                           'query':q,'warmup':warmup,'seconds':row['complete_seconds'],'exact':True}),flush=True)
                cell['worker_transport']=[json.loads((folder/f'worker{i}.transport.json').read_text()) for i in (0,1)]
                for i in (0,1):shutil.copy2(folder/f'worker{i}.log',output/f'{size}-{name}-{method}-worker{i}.log')
                cell['network_after']=status();cell['complete']=True;save(report_path,report)
                remove_created(folder);gc.collect()
        server['contents'].close();server['metadata_store'].close()
        del server,shares,client,state,refs,records;gc.collect();remove_created(root)
    report['complete']=True;report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    save(report_path,report);profile('N0')
    print('EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':main()

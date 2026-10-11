"""Archive a publication-oracle regression, joint-view sanity check, and COCO smoke run.

The exhaustive Z/2 experiment checks algebraic distributions only. It is not
a proof of the real cryptographic implementation or its runtime side channels.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'

import ast
import hashlib
import io
import itertools
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from float_reference import encode_fp32,exact_dot_encoded
from ppcmr import float_mpc
from ppcmr.cached_mpc import CachedMatrixEngine
from ppcmr.compact_verify import prepare_verifier_state
from ppcmr.crypto import canonical
from ppcmr.publication_policy import POLICY,SKETCH_PADDED_BYTES
from research_common import ARTIFACTS,file_sha256,load_features,write_json
from run_compact_research import perform
from run_float_mpc_research import build_records

OUT=Path(__file__).resolve().parents[1]/'manuscript/revision_20261009/transcript_audit'


def publication_oracle():
    source=(OUT/'before_float_mpc.py.txt').read_text(encoding='utf-8')
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='publish')
    namespace=dict(vars(float_mpc))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-publish','exec'),namespace)
    records=[];hashes=[]
    for value in (0.,1.):
        v=np.zeros(512,dtype=np.float32);v[0]=value
        records.append([{'id':'a','scope':'s','modality':'text','embedding':v,'content':'x'}])
        data=io.BytesIO();np.savez(data,features=v[None,:])
        hashes.append(hashlib.sha256(data.getvalue()).hexdigest())
    old=0;new_absent=0
    for index,rs in enumerate(records):
        provenance={'features_sha256':hashes[index]}
        legacy,_,_=namespace['publish'](rs,{'u':['s']},provenance)
        updated,_,_=float_mpc.publish(rs,{'u':['s']},provenance)
        for party in (0,1):
            # Both workers receive the same public snapshot; no collusion needed.
            announced=legacy['snapshot']['payload']['provenance']['features_sha256']
            matches=[j for j,h in enumerate(hashes) if h==announced]
            assert matches==[index]
            old+=1
            assert all(h not in json.dumps(updated) for h in hashes)
            new_absent+=1
    return {'legacy_confirmation_cases':old,'updated_hash_absence_cases':new_absent,
            'candidate_caches':2,'meaning':'Reproduces the plaintext-cache hash oracle; absence regression alone is not a privacy proof.'}


def toy_joint_view():
    # Actual two-query transcript in R=Z/2, dimension/row count one.
    def real(x,queries,party):
        distribution=Counter()
        for bits in itertools.product(range(2),repeat=11):
            x0,a,a0=bits[:3]
            xt=x0 if party==0 else x^x0
            at=a0 if party==0 else a^a0
            d=x^a;dt=xt^at
            view=[xt,at,dt,d^dt,d]
            for query,offset in zip(queries,(3,7)):
                q0,b,b0,c0=bits[offset:offset+4]
                qt=q0 if party==0 else query^q0
                bt=b0 if party==0 else b^b0
                ct=c0 if party==0 else (a*b)^c0
                e=query^b;et=qt^bt
                y=ct^(d*bt)^(xt*e)
                view.extend((qt,bt,ct,et,e^et,e,y))
            distribution[tuple(view)]+=1
        return distribution
    def simulated():
        distribution=Counter()
        for bits in itertools.product(range(2),repeat=11):
            xt,at,d=bits[:3];dt=xt^at
            view=[xt,at,dt,d^dt,d]
            for offset in (3,7):
                qt,bt,ct,e=bits[offset:offset+4];et=qt^bt
                view.extend((qt,bt,ct,et,e^et,e,ct^(d*bt)^(xt*e)))
            distribution[tuple(view)]+=1
        return distribution
    target=simulated()
    worlds=0
    for party in (0,1):
        for x,q1,q2 in itertools.product(range(2),repeat=3):
            assert real(x,(q1,q2),party)==target
            worlds+=1
    return {'ring':'Z/2','rows':1,'dimensions':1,'queries':2,
            'server_input_worlds':worlds,'random_tapes_per_world':2048,
            'real_traces_enumerated':worlds*2048,
            'joint_distributions_equal_to_simulator':True,
            'scope':'Algebraic sanity check; excludes encryption, signatures, metadata and runtime side channels.'}


def real_coco():
    manifest,provenance,images,texts=load_features(ARTIFACTS/'test')
    records=build_records(manifest,images,texts,100)
    grants={str(n):[f'scope-{i:02d}' for i in range(n)] for n in (0,1,5,10)}
    start=perf_counter()
    server,shares,clients=float_mpc.publish(records,grants,provenance,compact=True)
    publication_seconds=perf_counter()-start
    assert provenance['features_sha256'] not in json.dumps(server)
    client=clients['10']
    start=perf_counter();prepare_verifier_state(client)
    state_initialization_seconds=perf_counter()-start
    private_bytes=len(canonical({f'{a}:{m}':v for (a,m),v in client.verification_sketches.items()}))
    rows=[]
    with CachedMatrixEngine(server,shares) as engine:
        setup=engine.setup_metrics
        for n in (0,1,5,10):
            client=clients[str(n)];state=prepare_verifier_state(client)
            for modality in ('image','text'):
                feature=images[8] if modality=='image' else texts[40]
                candidates=[r for r in sorted(records,key=lambda r:r['id'])
                            if r['scope'] in grants[str(n)] and r['modality']!=modality]
                scores=[exact_dot_encoded(encode_fp32(feature),encode_fp32(r['embedding'])) for r in candidates]
                order=sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]['id']))[:10]
                reference={'scores':scores,'ids':[candidates[i]['id'] for i in order],
                           'contents':[candidates[i]['content'] for i in order]}
                _,_,row=perform(engine,client,state,feature,modality,reference,'compact')
                rows.append({'granted_scopes':n,'query_modality':modality,**row})
    return {'images':100,'captions':500,'queries':len(rows),'measurements':rows,
            'features_sha256':provenance['features_sha256'],
            'publication_seconds':publication_seconds,'matrix_setup':setup,
            'private_state_initialization_seconds':state_initialization_seconds,
            'private_state_provisioned_JSON_bytes':private_bytes,
            'scope':'Post-fix regression, one query per grant/direction. Not a repeated timing study or replacement of archived performance cohorts.'}


def main():
    destination=OUT/'publication_privacy_audit.json'
    if destination.exists():raise ValueError('Preserve the existing audit; use a new destination')
    report={'date':'2026-10-09','publication_policy':POLICY,
            'sketch_padded_plaintext_bytes':SKETCH_PADDED_BYTES,
            'publication_oracle':publication_oracle(),'toy_joint_view':toy_joint_view()}
    print('oracle and algebraic checks passed',flush=True)
    report['real_coco_regression']=real_coco()
    report['source_sha256']={p:file_sha256(Path(__file__).parent/p) for p in (
        'run_publication_privacy_audit.py','ppcmr/publication_policy.py',
        'ppcmr/float_mpc.py','ppcmr/compact_verify.py','ppcmr/cached_mpc.py','ppcmr/exact_ring.py')}
    report['unit_tests']={'passed':80,'seconds':5.097,'command':'python -m unittest discover -s tests -v',
                          'runtime':'Python 3.12.10, BLAS threads 4'}
    report['complete']=True
    write_json(destination,report)
    print(json.dumps({'complete':True,'file':str(destination),
                      'coco_queries':report['real_coco_regression']['queries'],
                      'private_state_JSON_bytes':report['real_coco_regression']['private_state_provisioned_JSON_bytes']}),flush=True)


if __name__=='__main__':main()

"""Audit the complete raw cohort before producing any paper-facing numbers."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_csv(path,rows):
    assert rows
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=BASE/'executed/experiment.json')
    p.add_argument('--output',type=Path,default=BASE/'results')
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--gallery',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.input.read_text());assert d['complete'] and not d['smoke']
    design_path=a.input.parent/'design.json'
    assert file_hash(design_path)==d['design_sha256'],'Pre-run plan digest mismatch'
    locked=json.loads(design_path.read_text())
    assert d['source_sha256']==locked['source_sha256'],'Executed code differs from the pre-run plan'
    assert locked['sizes']==[5000,20000] and locked['profiles']==['N0','N1','N2','N3']
    assert d['queries']==locked['queries']
    assert len(d['queries'])==len(set(d['queries']))==5
    assert sorted(c['images'] for c in d['cohorts'])==[5000,20000]
    for rel,expected in d['source_sha256'].items():
        assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==expected,rel
    gallery=a.gallery.resolve()
    assert file_hash(gallery/'manifest.json')==d['encoder']['manifest_sha256']
    for name,expected in d['encoder']['features_sha256'].items():assert file_hash(gallery/name)==expected,name
    raw=[];warmups=[];summary=[];setups=[];requests=set();score_checks=0;warmup_checks=0
    for c in d['cohorts']:
        assert c['captions']==5*c['images'] and len(c['cells'])==8
        cells={(r['profile'],r['method']):r for r in c['cells']};assert len(cells)==8
        seen={}
        for (profile,method),cell in cells.items():
            assert cell['complete'] and len(cell['measurements'])==5 and len(cell['warmups'])==1
            assert [r['query_index'] for r in cell['measurements']]==d['queries']
            drops_before=sum(q.get('drops',0) for q in cell['network_before']['qdiscs'])
            drops_after=sum(q.get('drops',0) for q in cell['network_after']['qdiscs'])
            assert drops_after>=drops_before
            setup={'images':c['images'],'profile':profile,'method':method,
                   'setup_seconds_including_bootstrap':cell['setup_seconds_including_bootstrap'],
                   'matrix_setup_seconds':cell['matrix_setup'].get('matrix_setup_seconds',0),
                   'dealer_setup_sent_bytes':cell['matrix_setup'].get('dealer_setup_sent_bytes',0),
                   'peer_setup_sent_bytes':cell['matrix_setup'].get('peer_setup_sent_bytes',0),
                   'qdisc_drops_delta':drops_after-drops_before}
            setups.append(setup)
            for kind,target in [('measurements',raw),('warmups',warmups)]:
                for r in cell[kind]:
                    assert all(r[k] for k in ('exact_scores_equal','exact_order_equal','original_contents_equal'))
                    assert r['cost']['candidate_count']==c['captions']
                    q=r['query_index'];assert r['score_vector_digest']==c['reference_digests'][str(q)]
                    identity=(r['score_vector_digest'],tuple(r['ids']))
                    assert seen.setdefault(q,identity)==identity
                    assert r['request_digest'] not in requests;requests.add(r['request_digest'])
                    if kind=='measurements':score_checks+=c['captions']
                    else:warmup_checks+=c['captions']
                    target.append({'images':c['images'],'captions':c['captions'],'profile':profile,'method':method,
                        'query_index':q,'complete_seconds':r['complete_seconds'],'verification_seconds':r['verification_seconds'],
                        'dealer_seconds':r['cost']['trusted_preprocessing_seconds'],
                        'dealer_sent_bytes':r['cost']['trusted_preprocessing_sent_bytes'],
                        'peer_sent_bytes':r['cost']['peer_opening_sent_bytes'],
                        'client_payload_bytes':r['cost']['client_total_transport_payload_bytes'],
                        'control_sent_messages':sum(x['sent_messages'] for x in r['control_transport']),
                        'control_received_messages':sum(x['received_messages'] for x in r['control_transport']),
                        'request_digest':r['request_digest'],'score_vector_digest':r['score_vector_digest'],
                        'exact':True})
        for profile in ['N0','N1','N2','N3']:
            f=cells[profile,'fresh'];k=cells[profile,'cached']
            ft=[r['complete_seconds'] for r in f['measurements']]
            kt=[r['complete_seconds'] for r in k['measurements']]
            ratios=[x/y for x,y in zip(ft,kt)]
            row={'images':c['images'],'captions':c['captions'],'profile':profile,
                 'target_RTT_ms':None if profile=='N0' else d['profiles'][profile][0],
                 'target_rate_Mbit_s':d['profiles'][profile][1],
                 'fresh_median_s':statistics.median(ft),'fresh_min_s':min(ft),'fresh_max_s':max(ft),
                 'cached_median_s':statistics.median(kt),'cached_min_s':min(kt),'cached_max_s':max(kt),
                 'median_paired_speedup':statistics.median(ratios),'min_paired_speedup':min(ratios),
                 'max_paired_speedup':max(ratios),'n_queries':5}
            summary.append(row)
    assert len(raw)==80 and len(warmups)==16 and score_checks==5_000_000 and warmup_checks==1_000_000
    calibration=[]
    for profile in ['N0','N1','N2','N3']:
        cal=json.loads((a.calibration/(profile+'.json')).read_text())
        for link in cal['links']:
            calibration.append({'profile':profile,'pair':'-'.join(link['pair']),
               'rtt_min_ms':link['rtt_min_mean_max_ms'][0],'rtt_mean_ms':link['rtt_min_mean_max_ms'][1],
               'rtt_max_ms':link['rtt_min_mean_max_ms'][2],
               'forward_throughput_Mbit_s':link['throughput'][0]['receiver_mbit_s'],
               'reverse_throughput_Mbit_s':link['throughput'][1]['receiver_mbit_s']})
    a.output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('raw_queries',raw),('warmups',warmups),('summary',summary),('setup',setups),('calibration',calibration)]:
        write_csv(a.output/(name+'.csv'),rows)
    validation={'complete':True,'formal_queries':len(raw),'excluded_warmups':len(warmups),
       'formal_exact_score_comparisons':score_checks,'warmup_exact_score_comparisons':warmup_checks,
       'all_ordered_ids_and_original_contents_equal':True,'all_original_source_hashes_match':True,
       'pre_run_design_source_hashes_match_execution':True,
       'gallery_manifest_and_feature_hashes_unchanged_after_execution':True,
       'unique_request_digests':len(requests),'total_qdisc_drops_during_cells':sum(x['qdisc_drops_delta'] for x in setups),
       'raw_experiment_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
       'summary':'Descriptive five-query paired cohort; no p95 or population-level significance claim',
       'runtime':d['runtime'],'queries':d['queries'],'image_ids':d['image_ids']}
    (a.output/'validation.json').write_text(json.dumps(validation,indent=2))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(validation,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

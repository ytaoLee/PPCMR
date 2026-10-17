"""Experiment-owned full-mesh TCP network; qdiscs are on router egress."""
import argparse
import json
import subprocess
from pathlib import Path

PREFIX = 'ppcmrn16'
NS = {k: PREFIX+'-'+k for k in ('h','c0','c1','r')}
LINKS = [
    ('h','h0','10.218.0.1','c0','ch','10.218.0.2','b0'),
    ('h','h1','10.218.1.1','c1','ch','10.218.1.2','b1'),
    ('c0','cp','10.218.2.1','c1','cp','10.218.2.2','b2'),
]
PROFILES = {'N0':(0,None),'N1':(20,1000),'N2':(80,1000),'N3':(80,100)}

def run(args, **kw):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,**kw)
    if p.returncode:raise RuntimeError(f'{args}: {p.stderr}')
    return p

def inside(node,*args):
    return run(['ip','netns','exec',NS[node],*args])

def create():
    existing={r['name'] for r in json.loads(run(['ip','-j','netns','list']).stdout or '[]')}
    assert not existing.intersection(NS.values()), 'Do not reuse or replace an existing namespace'
    for key in ('rmem_max','wmem_max'):
        run(['sysctl','-w','net.core.'+key+'=33554432'])
    for name in NS.values():
        run(['ip','netns','add',name])
        run(['ip','-n',name,'link','set','lo','up'])
    for i,(a,da,ipa,b,db,ipb,bridge) in enumerate(LINKS):
        inside('r','ip','link','add',bridge,'type','bridge')
        inside('r','ip','link','set',bridge,'up')
        for j,(node,dev,ipaddr) in enumerate(((a,da,ipa),(b,db,ipb))):
            temp=f'pn{i}{j}';router=f'r{i}{j}'
            run(['ip','link','add',temp,'type','veth','peer','name',router])
            run(['ip','link','set',temp,'netns',NS[node]])
            run(['ip','link','set',router,'netns',NS['r']])
            inside(node,'ip','link','set',temp,'name',dev)
            inside(node,'ip','address','add',ipaddr+'/30','dev',dev)
            inside(node,'ip','link','set',dev,'up')
            inside('r','ip','link','set',router,'master',bridge)
            inside('r','ip','link','set',router,'up')
            # Keep virtual aggregation from turning the rate shaper into a
            # coarse large-segment timer. All devices belong to this experiment.
            for target,device in ((node,dev),('r',router)):
                inside(target,'ethtool','-K',device,'tso','off','gso','off','gro','off')
    for node in ('h','c0','c1'):
        for key,value in [('net.ipv4.tcp_rmem','4096 131072 33554432'),
                          ('net.ipv4.tcp_wmem','4096 16384 33554432')]:
            inside(node,'sysctl','-w',key+'='+value)
    profile('N0')

def profile(name):
    rtt,rate=PROFILES[name]
    for i in range(3):
        for j in range(2):
            dev=f'r{i}{j}'
            args=['tc','qdisc','replace','dev',dev,'root']
            if rate:
                args+=['netem','limit','20000','delay',f'{rtt/2:g}ms','rate',f'{rate}mbit']
            else:args+=['pfifo','limit','20000']
            inside('r',*args)

def status():
    return {'namespaces':NS,'links':LINKS,'qdiscs':json.loads(inside('r','tc','-s','-j','qdisc','show').stdout)}

def cleanup():
    # Only exact names created by this script, never the host's interfaces.
    existing={r['name'] for r in json.loads(run(['ip','-j','netns','list']).stdout or '[]')}
    for name in NS.values():
        if name in existing:run(['ip','netns','delete',name])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','profile','status','cleanup'])
    p.add_argument('--profile',choices=PROFILES,default='N0');a=p.parse_args()
    if a.action=='create':create()
    elif a.action=='profile':profile(a.profile)
    elif a.action=='cleanup':cleanup()
    if a.action!='cleanup':print(json.dumps(status()))

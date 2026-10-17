"""Measured RTT and bidirectional single-flow throughput for every test link."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from network_topology import NS,LINKS,PROFILES,profile,status

def call(node,args):
    return subprocess.run(['ip','netns','exec',NS[node],*args],capture_output=True,text=True,check=True)

def calibrate(name):
    profile(name);rows=[]
    for index,(a,da,ipa,b,db,ipb,bridge) in enumerate(LINKS):
        ping=call(a,['ping','-c','12','-i','0.1',ipb]).stdout
        match=re.search(r'= ([\d.]+)/([\d.]+)/([\d.]+)',ping)
        assert match,ping
        rates=[]
        for sender,receiver,address in ((a,b,ipb),(b,a,ipa)):
            port=46000+index
            server=subprocess.Popen(['ip','netns','exec',NS[receiver],'iperf3','-s','-1','-p',str(port),'-J'],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            time.sleep(.15)
            client=call(sender,['iperf3','-c',address,'-p',str(port),'-t','4','-O','1','-w','8M','-J'])
            sout,serr=server.communicate(timeout=10);assert server.returncode==0,serr
            result=json.loads(client.stdout);assert 'error' not in result,result
            rates.append({'sender':sender,'receiver':receiver,'client':result,'server':json.loads(sout),
                          'receiver_mbit_s':result['end']['sum_received']['bits_per_second']/1e6})
        rows.append({'pair':[a,b],'ping_stdout':ping,'rtt_min_mean_max_ms':list(map(float,match.groups())),
                     'throughput':rates})
        print(json.dumps({'calibration':name,'pair':[a,b],'rtt_ms':rows[-1]['rtt_min_mean_max_ms'],
                          'mbit_s':[r['receiver_mbit_s'] for r in rates]}),flush=True)
    return {'profile':name,'target':PROFILES[name],'links':rows,'network_after':status()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name in PROFILES:
        result=calibrate(name)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2))
    profile('N0')

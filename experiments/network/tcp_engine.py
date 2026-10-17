"""TCP adapter: reuse the frozen streamed protocol and its worker unchanged."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import argparse
import json
import os
import pickle
import select
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from time import perf_counter
from ppcmr.streaming_mpc import StreamingEngine,_stream_worker
from ppcmr.crypto import digest,load_public_key,verify_signature
from ppcmr.publication_policy import validate_publication
from network_topology import NS

SCRIPT=Path(__file__).resolve()

def tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,16*1024*1024)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,16*1024*1024)
    sock.settimeout(600)
    return sock

class TCPConnection:
    def __init__(self,sock):
        self.sock=tune(sock);self.sent_messages=0;self.received_messages=0
        self.sent_bytes=0;self.received_bytes=0
    def send_bytes(self,data):
        self.sock.sendall(struct.pack('!Q',len(data)))
        self.sock.sendall(data)
        self.sent_messages+=1;self.sent_bytes+=len(data)+8
    def _exact(self,count):
        data=bytearray(count);view=memoryview(data);offset=0
        while offset<count:
            n=self.sock.recv_into(view[offset:])
            if not n:raise EOFError('TCP peer closed mid-frame')
            offset+=n
        return data
    def recv_bytes(self):
        size=struct.unpack('!Q',self._exact(8))[0]
        if size>64*1024*1024:raise ValueError('Unexpectedly large experiment frame')
        result=self._exact(size);self.received_messages+=1;self.received_bytes+=size+8
        return bytes(result)
    def poll(self,timeout=0):return bool(select.select([self.sock],[],[],timeout)[0])
    def close(self):self.sock.close()
    def counters(self):
        return {k:getattr(self,k) for k in ('sent_messages','received_messages','sent_bytes','received_bytes')}

def listener(address):
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(address);sock.listen(2);sock.settimeout(60);return sock

def connect(address):
    deadline=time.monotonic()+60
    while True:
        sock=socket.socket();tune(sock)
        try:sock.connect(address);return TCPConnection(sock)
        except ConnectionRefusedError:
            sock.close()
            if time.monotonic()>deadline:raise
            time.sleep(.05)

class ProcessAdapter:
    def __init__(self,process):self.process=process;self.pid=process.pid
    def is_alive(self):return self.process.poll() is None
    def join(self,timeout):
        try:self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass
    def terminate(self):self.process.terminate()

class TCPEngine(StreamingEngine):
    def __init__(self,server_data,shares,folder,mode='cached',block_rows=256):
        assert mode in {'fresh','cached'} and block_rows==256
        snap=verify_signature(load_public_key(server_data['owner_public']),server_data['snapshot'],'float-snapshot')
        validate_publication(snap)
        self.server_data=server_data;self._snapshot=digest(server_data['snapshot'])
        self.mode=mode;self.block_rows=block_rows
        self.positions={m['id']:i for i,m in enumerate(server_data['records'])}
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.connections=[];self.processes=[];self.logs=[]
        listening=[listener(('10.218.0.1',45000)),listener(('10.218.1.1',45001))]
        start=perf_counter()
        for party in (0,1):
            spec=self.folder/f'worker{party}.pickle'
            with spec.open('wb') as f:pickle.dump({'party':party,'server':server_data,'share':shares[party],
                'folder':str(self.folder),'mode':mode,'block_rows':block_rows},f,protocol=5)
            os.chmod(spec,0o600)
            log=(self.folder/f'worker{party}.log').open('wb');self.logs.append(log)
            proc=subprocess.Popen(['ip','netns','exec',NS[f'c{party}'],sys.executable,str(SCRIPT),
                '--worker',str(spec)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            self.processes.append(ProcessAdapter(proc))
        for sock in listening:
            connection,_=sock.accept();self.connections.append(TCPConnection(connection));sock.close()
        self.startup_seconds=perf_counter()-start;self.received_bytes=0
        self.matrix_context=None;self.setup_metrics={}
        if mode=='cached':self._setup_cache()
    def counters(self):return [c.counters() for c in self.connections]
    def close(self):
        super().close()
        for stream in self.logs:stream.close()

def worker(path):
    with open(path,'rb') as f:s=pickle.load(f)
    party=s['party'];control=connect((f'10.218.{party}.1',45000+party))
    if party==0:
        sock=listener(('10.218.2.1',45002));peer=TCPConnection(sock.accept()[0]);sock.close()
    else:peer=connect(('10.218.2.1',45002))
    try:_stream_worker(party,control,peer,s['server'],s['share'],s['mode'],s['folder'],s['block_rows'])
    finally:
        (Path(s['folder'])/f'worker{party}.transport.json').write_text(json.dumps(
            {'control':control.counters(),'peer':peer.counters()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();worker(a.worker)

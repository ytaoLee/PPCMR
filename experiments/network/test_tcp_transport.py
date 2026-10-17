"""Run in the experiment Linux environment before a new measurement cohort."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
import os
import socket
import threading
from tcp_engine import TCPConnection,connect,listener

def main():
    server=listener(('127.0.0.1',0));address=server.getsockname()
    payloads=[b'',b'context-binding',os.urandom(6*1024*1024)];observed=[];errors=[]
    def receive():
        try:
            peer=TCPConnection(server.accept()[0])
            for expected in payloads:
                actual=peer.recv_bytes()
                assert isinstance(actual,bytes) and actual==expected
                peer.send_bytes(actual)
            observed.append(peer.counters());peer.close()
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=receive);thread.start();client=connect(address)
    for value in payloads:
        client.send_bytes(value);assert client.poll(10)
        received=client.recv_bytes();assert isinstance(received,bytes) and received==value
    thread.join(timeout=10);assert not thread.is_alive() and not errors,errors
    expected=sum(len(p)+8 for p in payloads)
    assert client.sent_bytes==client.received_bytes==expected
    assert observed[0]['sent_bytes']==observed[0]['received_bytes']==expected
    client.close();server.close();print('TCP frame/type/count/empty/large-payload tests PASS')

if __name__=='__main__':main()

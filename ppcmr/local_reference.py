"""Equivalent authorized feature-download + client exact search baseline.

Each query downloads a fresh complete authenticated target collection. No
query feature or query share is sent to this worker. A trusted snapshot and
scope keys are already provisioned, as in the MPC baseline. Private local IPC
is used; payload byte counts exclude pipe framing. Feature caching across
queries is deliberately disabled and is a potential further local saving.
"""

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from time import perf_counter

from .crypto import (canonical, decrypt_content, digest, load_public_key,
                     sign, verify_signature)
from .float_mpc import (_recv, _recv_counted, _send, aad, authenticate,
                        make_query_context, require, validate_feature,
                        validate_witness)
from float_reference import exact_dot_encoded
from .crypto import decode
import numpy as np


@dataclass
class LocalQuerySession:
    request: dict
    feature: np.ndarray


def make_local_query(client, feature, modality, k=10):
    validate_feature(feature)
    return LocalQuerySession(make_query_context(client, modality, k), feature.copy())


def _local_worker(control, server_data):
    used_nonces, sessions = set(), {}
    while True:
        message = _recv(control)
        if message.get('command') == 'stop':
            break
        try:
            if message['command'] == 'prepare':
                _, cred, req = authenticate(server_data['owner_public'], server_data['snapshot'],
                                           message['credential'], message['request'])
                require(req['nonce'] not in used_nonces, 'Replayed query')
                used_nonces.add(req['nonce'])
                witnesses = [m for m in server_data['records']
                             if m['scope'] in cred['allowed_scopes'] and m['modality'] == req['target_modality']]
                identifier = digest(message['request'])
                sessions[identifier] = (cred, {m['id'] for m in witnesses})
                _send(control, {'ok':True, 'witnesses':witnesses, 'pid':os.getpid()})
            elif message['command'] == 'fetch':
                cred, allowed = sessions[message['request_digest']]
                require(cred['expires_at'] > time.time(), 'Expired credential')
                payload = verify_signature(load_public_key(cred['query_public']), message['fetch'], 'float-fetch')
                require(set(payload) == {'request_digest','ids'} and
                        payload['request_digest'] == message['request_digest'], 'Wrong fetch request')
                ids = payload['ids']
                require(isinstance(ids,list) and len(set(ids)) == len(ids) and set(ids).issubset(allowed),
                        'Unauthorized content fetch')
                _send(control, {'ok':True, 'results':[{'id':i,'content':server_data['contents'][i]} for i in ids]})
            else:
                raise ValueError('Unknown local-baseline command')
        except Exception as error:
            _send(control, {'ok':False, 'error':str(error)})
    control.close()


class AuthorizedLocalEngine:
    def __init__(self, server_data):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe(duplex=True)
        self.process = context.Process(target=_local_worker,args=(child,server_data))
        started = perf_counter()
        self.process.start()
        child.close()
        self.startup_seconds = perf_counter()-started

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def close(self):
        if self.process.is_alive():
            _send(self.connection, {'command':'stop'})
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.connection.close()

    def _receive(self):
        require(self.connection.poll(60), 'Local baseline response timeout')
        response, size = _recv_counted(self.connection)
        require(response['ok'], response.get('error','Local baseline failure'))
        return response, size

    def search(self, client, session):
        started = perf_counter()
        sent = _send(self.connection, {'command':'prepare','request':session.request,'credential':client.credential})
        prepared, received = self._receive()
        collection_seconds = perf_counter()-started
        local_started = perf_counter()
        witnesses = prepared['witnesses']
        snap, _, req = validate_witness(client,session,witnesses)
        query = validate_feature(session.feature)
        scores = []
        for m in witnesses:
            data = decode(decrypt_content(client.scope_keys[m['scope']],m['feature'],aad(snap,m,'feature')))
            require(len(data)==512*4, 'Malformed encrypted embedding')
            feature = np.frombuffer(data,dtype='<f4').copy()
            scores.append(exact_dot_encoded(query,validate_feature(feature)))
        order = sorted(range(len(witnesses)),key=lambda i:(-scores[i],witnesses[i]['id']))[:req['k']]
        selected = [witnesses[i] for i in order]
        local_seconds = perf_counter()-local_started
        identifier = digest(session.request)
        fetch = sign(client.signing_key, {'request_digest':identifier,'ids':[m['id'] for m in selected]}, 'float-fetch')
        fetch_started = perf_counter()
        sent += _send(self.connection, {'command':'fetch','request_digest':identifier,'fetch':fetch})
        fetched, size = self._receive()
        received += size
        require([r['id'] for r in fetched['results']] == [m['id'] for m in selected], 'Incorrect fetched results')
        contents = []
        for r,m in zip(fetched['results'],selected):
            require(set(r)=={'id','content'} and digest(r['content'])==m['content_digest'], 'Altered content')
            contents.append(decrypt_content(client.scope_keys[m['scope']],r['content'],aad(snap,m,'content')))
        fetch_seconds = perf_counter()-fetch_started
        # Scores originate locally and are not part of the cloud's response.
        result = {'request_digest':identifier,'snapshot_digest':digest(client.snapshot),
                  'witnesses':witnesses,'results':fetched['results']}
        response_json_bytes = len(canonical(result))
        return {'accepted':True,'ids':[m['id'] for m in selected], 'scores':scores, 'contents':contents}, {
            'collection_delivery_seconds':collection_seconds,
            'authenticate_decrypt_exact_rank_seconds':local_seconds,
            'content_fetch_check_seconds':fetch_seconds,
            'search_seconds_including_serialization':perf_counter()-started,
            'client_request_sent_bytes':sent,'client_response_received_bytes':received,
            'client_total_transport_payload_bytes':sent+received,
            'result_json_bytes':response_json_bytes,'candidate_count':len(witnesses),
            'server_process_id':prepared['pid']}

"""Bounded ring-file windows and streamed authenticated publication.

Role-private files replace dense Python integer matrices. Opened masked values
are still exchanged through IPC; file paths do not substitute for protocol
delivery. The local launcher is trusted, as in the existing same-host harness.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter
import numpy as np

from .crypto import (canonical,digest,encode,encrypt_content,export_public_key,
                     new_owner_key,sign,merkle_root)
from .compact_verify import coefficient,sketch_context,PRIME,make_catalog,FORMAT
from .float_mpc import DIM,BITS,MASK,FloatClient,aad,validate_feature,require,pack_words
from .publication_policy import POLICY,SNAPSHOT_SCHEMA,FEATURE_PROFILE,SKETCH_PADDED_BYTES

LIMBS=20
ROW_BYTES=DIM*LIMBS*2


def random_digits(rows):
    return np.frombuffer(os.urandom(rows*ROW_BYTES),dtype='<u2').reshape(rows,DIM,LIMBS)


def ring_combine(left,right,subtract=False):
    require(left.shape==right.shape and left.dtype==right.dtype==np.dtype('<u2'),'Invalid ring operands')
    out=np.empty_like(left);carry=np.zeros(left.shape[:-1],dtype=np.int32)
    for limb in range(LIMBS):
        value=left[...,limb].astype(np.int32)+carry
        value=value-right[...,limb] if subtract else value+right[...,limb]
        out[...,limb]=(value & 65535).astype('<u2');carry=value>>16
    return out


def digits_words(matrix):
    return [int.from_bytes(row.tobytes(),'little') for row in matrix.reshape(-1,LIMBS)]


class RingFile:
    """Read only selected row windows, without retaining mapped resident pages."""
    def __init__(self,path,rows,create=False,dtype='<u2',limbs=20):
        self.path=str(Path(path).resolve());self.rows=rows;self.dtype=np.dtype(dtype);self.limbs=limbs
        self.row_bytes=DIM*limbs*self.dtype.itemsize
        if create:
            with open(self.path,'xb') as stream:stream.truncate(rows*self.row_bytes)
        require(Path(self.path).stat().st_size==rows*self.row_bytes,'Ring file dimensions differ')

    def write(self,start,matrix):
        require(matrix.dtype==self.dtype and matrix.shape[1:]==(DIM,self.limbs), 'Wrong ring window')
        require(0<=start<=start+len(matrix)<=self.rows,'Ring write outside file')
        with open(self.path,'r+b') as stream:
            stream.seek(start*self.row_bytes);stream.write(matrix.tobytes())

    def read(self,indices):
        indices=list(indices)
        require(all(type(i) is int and 0<=i<self.rows for i in indices),'Ring read outside file')
        if not indices:return np.empty((0,DIM,self.limbs),dtype=self.dtype)
        # Coalesce consecutive selected rows. Only the selected window is held.
        pieces=[];start=0
        with open(self.path,'rb') as stream:
            while start<len(indices):
                end=start+1
                while end<len(indices) and indices[end]==indices[end-1]+1:end+=1
                stream.seek(indices[start]*self.row_bytes)
                blob=stream.read((end-start)*self.row_bytes)
                require(len(blob)==(end-start)*self.row_bytes,'Truncated ring window')
                pieces.append(np.frombuffer(blob,dtype=self.dtype).reshape(end-start,DIM,self.limbs))
                start=end
        return pieces[0] if len(pieces)==1 else np.concatenate(pieces)


class DiskValues:
    def __init__(self,path,field):self.path=str(Path(path).resolve());self.field=field;self.connection=None
    def _db(self):
        if self.connection is None:self.connection=sqlite3.connect(self.path)
        return self.connection
    def __getitem__(self,identifier):
        row=self._db().execute('SELECT '+self.field+' FROM items WHERE id=?',(identifier,)).fetchone()
        if row is None:raise KeyError(identifier)
        return json.loads(row[0])
    def __getstate__(self):return {'path':self.path,'field':self.field,'connection':None}
    def close(self):
        if self.connection is not None:self.connection.close();self.connection=None


def root_from_leaves(leaves):
    if not leaves:return hashlib.sha256(b'ppcmr:empty:v1').hexdigest()
    level=list(leaves)
    while len(level)>1:
        if len(level)%2:level.append(level[-1])
        level=[hashlib.sha256(b'ppcmr:node:v1\x00'+level[i]+level[i+1]).digest()
               for i in range(0,len(level),2)]
    return level[0].hex()


def publish_streamed(records,grants,folder,block_rows=256,epoch=1,database='coco-streamed'):
    started=perf_counter();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    require(len(records)>0 and 1<=block_rows<=4096,'Invalid publication size/block')
    owner=new_owner_key();public=export_public_key(owner.public_key())
    scopes=sorted({r['scope'] for r in records.descriptors})
    keys={s:os.urandom(32) for s in scopes};header={'database':database,'epoch':epoch}
    parts={(s,m):{'leaves':[],'seed':os.urandom(32),'columns':[0]*DIM,'catalog':[]}
           for s in scopes for m in ('image','text')}
    share_files=[RingFile(folder/f'X_{party}.bin',len(records),create=True) for party in (0,1)]
    db_path=folder/'encrypted.sqlite';require(not db_path.exists(),'Publication file already exists')
    db=sqlite3.connect(db_path);db.execute('PRAGMA journal_mode=OFF')
    db.execute('CREATE TABLE items (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, content TEXT NOT NULL)')
    catalog=[];sketch_seconds=0.;previous=None
    for begin in range(0,len(records),block_rows):
        encoded=[]
        for position in range(begin,min(begin+block_rows,len(records))):
            r=records[position]
            require(previous is None or previous<r['id'],'Noncanonical or duplicate record order');previous=r['id']
            values=validate_feature(r['embedding']);encoded.append([x&MASK for x in values])
            m={k:r[k] for k in ('id','scope','modality')}
            m['feature']=encrypt_content(keys[r['scope']],encode(r['embedding'].astype('<f4',copy=False).tobytes()),aad(header,m,'feature'))
            content=encrypt_content(keys[r['scope']],r['content'],aad(header,m,'content'))
            m['content_digest']=digest(content);c=make_catalog(m);catalog.append(c)
            db.execute('INSERT INTO items VALUES (?,?,?)',(m['id'],canonical(m).decode('utf-8'),canonical(content).decode('utf-8')))
            p=parts[(r['scope'],r['modality'])]
            p['leaves'].append(hashlib.sha256(b'ppcmr:leaf:v1\x00'+canonical(m)).digest());p['catalog'].append(c)
            t=perf_counter();weight=coefficient(p['seed'],sketch_context(header,r['scope'],r['modality']),r['id'])
            p['columns']=[(a+weight*x)%PRIME for a,x in zip(p['columns'],values)]
            sketch_seconds+=perf_counter()-t
        matrix=np.frombuffer(pack_words([v for row in encoded for v in row]),dtype='<u2').reshape(len(encoded),DIM,LIMBS)
        first=random_digits(len(matrix));second=ring_combine(matrix,first,True)
        share_files[0].write(begin,first);share_files[1].write(begin,second)
    db.commit();db.close()
    partitions=[];sketch_blobs={}
    for (scope,modality),p in parts.items():
        context=sketch_context(header,scope,modality);catalog_root=merkle_root(p['catalog'])
        data={'schema':FORMAT,'catalog_root':catalog_root,'count':len(p['catalog']),
              'field_bits':521,'seed':encode(p['seed']),'columns':p['columns']}
        plaintext=json.dumps(data,separators=(',',':'));require(len(plaintext)<=SKETCH_PADDED_BYTES,'Sketch exceeds padding')
        blob=encrypt_content(keys[scope],plaintext+' '*(SKETCH_PADDED_BYTES-len(plaintext)),context)
        sketch_blobs[(scope,modality)]=blob
        partitions.append({'scope':scope,'modality':modality,'count':len(p['catalog']),
            'root':root_from_leaves(p['leaves']),'catalog_root':catalog_root,'sketch_digest':digest(blob)})
    snapshot=sign(owner,{**header,'schema':SNAPSHOT_SCHEMA,'dimensions':DIM,'ring_bits':BITS,
        'feature_scale_exponent':149,'score_scale_exponent':298,'ranking':'exact score desc, ID asc',
        'publication_policy':POLICY,'public_feature_profile':dict(FEATURE_PROFILE),'partitions':partitions,
        'compact_verification':FORMAT,'private_sketch_plaintext_bytes':SKETCH_PADDED_BYTES},'float-snapshot')
    clients={}
    for subject,allowed in grants.items():
        allowed=sorted(set(allowed));require(set(allowed).issubset(scopes),'Unknown permission scope')
        signing=new_owner_key();credential=sign(owner,{'schema':'ppcmr.float-mpc.credential.v1','subject':subject,
            'snapshot_digest':digest(snapshot),'allowed_scopes':allowed,'query_public':export_public_key(signing.public_key()),
            'expires_at':int(__import__('time').time())+7200},'float-credential')
        clients[subject]=FloatClient(public,snapshot,credential,{s:keys[s] for s in allowed},signing,
            {k:v for k,v in sketch_blobs.items() if k[0] in allowed})
    server={'owner_public':public,'snapshot':snapshot,'records':catalog,'catalog':catalog,
            'contents':DiskValues(db_path,'content'),'metadata_store':DiskValues(db_path,'metadata')}
    metrics={'publication_seconds':perf_counter()-started,'private_sketch_generation_seconds':sketch_seconds,
             'records':len(records),'block_rows':block_rows,'encrypted_disk_bytes':db_path.stat().st_size,
             'two_share_file_bytes':sum(Path(s.path).stat().st_size for s in share_files)}
    return server,share_files,clients,metrics

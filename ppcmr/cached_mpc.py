"""Fixed-snapshot matrix-mask caching, with fresh query masks and score shares.

A is an input-independent uniform matrix fixed for immutable X. D=X-A opens
once. Every request uses independent uniform b and a fresh split of c=A*b.
Summing c_t+D*b_t+X_t*E, E=q-b, yields X*q in the unchanged 320-bit ring.
The dealer keeps A, never D; masks MUST rotate when the published X changes.
Cold setup traffic and CPU are reported separately, never counted as zero.
"""
import os
from time import perf_counter

from .crypto import digest
from .exact_ring import matrix_digits, matvec
from .float_mpc import (BITS,DIM,MASK,TwoCloudEngine,_send,pack_words,
                        random_words,require,split_values)


def cached_material(matrix, indices, context, kernel='python', selected_digits=None):
    """Fresh input-independent (b,A*b) correlation for authorized fixed rows."""
    require(kernel in {'python','limb-fp64'},'Unsupported exact ring kernel')
    n=len(indices)
    b=random_words(DIM) if n else []
    if n and kernel=='limb-fp64':
        require(selected_digits is not None,'Missing selected matrix limbs')
        c=matvec(selected_digits,b)
    else:
        c=[sum(x*y for x,y in zip(matrix[i],b))&MASK for i in indices]
    b0,b1=split_values(b); c0,c1=split_values(c)
    identifier=os.urandom(16).hex()
    return tuple({'id':identifier,'mask':context['mask'],'snapshot':context['snapshot'],
                  'b_blob':pack_words(bb),'c_blob':pack_words(cc)}
                 for bb,cc in ((b0,c0),(b1,c1)))


class CachedMatrixEngine(TwoCloudEngine):
    compute_command='compute_cached'

    def __init__(self,server_data,shares,kernel='limb-fp64'):
        require(kernel in {'python','limb-fp64'},'Unsupported exact ring kernel')
        self.kernel=kernel
        self._snapshot_digest=digest(server_data['snapshot'])
        self._positions={m['id']:i for i,m in enumerate(server_data['records'])}
        self._selected_digits=None
        cold_started=perf_counter()
        super().__init__(server_data,shares)
        self.received_bytes=0
        try:
            started=perf_counter()
            n=len(server_data['records'])
            flat=random_words(n*DIM)
            first,second=split_values(flat)
            self._matrix=[flat[i:i+DIM] for i in range(0,len(flat),DIM)]
            self._matrix_digits=matrix_digits(self._matrix)
            self.matrix_context={'stage':'immutable-matrix-open.v1','mask':os.urandom(16).hex(),
                'snapshot':self._snapshot_digest,'record_order':digest(list(self._positions)),
                'rows':n,'dimensions':DIM,'ring_bits':BITS}
            generation_seconds=perf_counter()-started
            started=perf_counter(); delivery_bytes=0
            for connection,own in zip(self.connections,(first,second)):
                delivery_bytes+=_send(connection,{'command':'cache_matrix','context':self.matrix_context,
                                                  'A_blob':pack_words(own)})
            responses=self._responses()
            require(all(r['ok'] for r in responses),'Matrix-cache initialization failed')
            require(responses[0]['opened_digest']==responses[1]['opened_digest'],
                    'Servers disagree on immutable masked matrix')
            self.setup_metrics={'cold_setup_seconds':perf_counter()-cold_started,
                'matrix_mask_generation_seconds':generation_seconds,
                'matrix_delivery_opening_seconds':perf_counter()-started,
                'dealer_setup_sent_bytes':delivery_bytes,
                'peer_setup_sent_bytes':sum(r['peer_sent_bytes'] for r in responses),
                'control_setup_received_bytes':self.received_bytes,'records':n,
                'one_mask_per_immutable_snapshot':True,'fresh_b_and_c_shares_per_query':True}
        except Exception:
            self.close()
            raise

    def _make_material(self,witnesses):
        indices=[self._positions[m['id']] for m in witnesses]
        digits=None
        if indices and self.kernel=='limb-fp64':
            key=tuple(indices)
            if self._selected_digits is None or self._selected_digits[0]!=key:
                self._selected_digits=(key,self._matrix_digits[indices])
            digits=self._selected_digits[1]
        return cached_material(self._matrix,indices,self.matrix_context,self.kernel,digits)

    def _compute_message(self,identifier,material):
        return {'command':self.compute_command,'request_digest':identifier,
                'triple':material,'kernel':self.kernel}

    def search(self,client,session,witness_mode='full'):
        require(digest(self.server_data['snapshot'])==self._snapshot_digest,
                'Immutable matrix cache requires new workers for a new snapshot')
        result,cost=super().search(client,session,witness_mode)
        cost.update({'mpc_mode':'immutable-matrix-cache.v1','ring_kernel':self.kernel,
                     'matrix_setup_excluded_from_query':True})
        return result,cost

"""Exact arithmetic in Z/(2^320), using bounded integer limbs in FP64 GEMM.

This does not approximate feature vectors or scores. Twenty unsigned 16-bit
limbs represent a ring word. Each 512-term limb dot product is an integer
below 2^41, hence ordinary IEEE binary64 products and accumulation are exact.
Coefficients and carries use uint64 below 2^46; the final 320-bit word is
assembled in Python. Requires genuine FP64 BLAS with no precision reduction.
"""
import numpy as np

BITS=320
DIM=512
LIMB_BITS=16
LIMBS=BITS//LIMB_BITS
BASE=1<<LIMB_BITS
MASK=(1<<BITS)-1


def matrix_digits(rows):
    if any(len(row)!=DIM for row in rows):
        raise ValueError('Expected 512-column ring matrix')
    if any(type(v) is not int or not 0<=v<=MASK for row in rows for v in row):
        raise ValueError('Invalid ring matrix word')
    blob=b''.join(v.to_bytes(BITS//8,'little') for row in rows for v in row)
    return np.frombuffer(blob,dtype='<u2').reshape(len(rows),DIM,LIMBS)


def matvec(matrix, vector):
    if (not isinstance(matrix,np.ndarray) or matrix.dtype!=np.dtype('<u2') or
            matrix.ndim!=3 or matrix.shape[1:]!=(DIM,LIMBS)):
        raise ValueError('Expected unsigned 16-bit matrix limbs')
    if len(vector)!=DIM or any(type(v) is not int or not 0<=v<=MASK for v in vector):
        raise ValueError('Invalid ring vector')
    rows=len(matrix)
    if rows==0:
        return []
    blob=b''.join(v.to_bytes(BITS//8,'little') for v in vector)
    right=np.frombuffer(blob,dtype='<u2').reshape(DIM,LIMBS).astype(np.float64)
    coefficients=np.zeros((rows,LIMBS),dtype=np.uint64)
    # A_i @ b_j contributes to digit i+j; digits >=20 vanish modulo 2^320.
    for i in range(LIMBS):
        left=np.ascontiguousarray(matrix[:,:,i],dtype=np.float64)
        products=left @ right[:,:LIMBS-i]
        if products.dtype!=np.float64:
            raise ValueError('FP64 kernel unavailable')
        coefficients[:,i:] += products.astype(np.uint64)
    carry=np.zeros(rows,dtype=np.uint64)
    output=np.zeros((rows,LIMBS),dtype='<u2')
    for i in range(LIMBS):
        value=coefficients[:,i]+carry
        output[:,i]=(value & np.uint64(BASE-1)).astype('<u2')
        carry=value>>np.uint64(LIMB_BITS)
    return [int.from_bytes(row.tobytes(),'little') for row in output]

"""Public, input-independent representation parameters for new publications.

Research provenance (including plaintext-cache hashes) stays in experiment
reports. It must never be copied into the protocol's signed public snapshot.
"""

POLICY = 'ppcmr.input-independent-publication.v2'
SNAPSHOT_SCHEMA = 'ppcmr.float-mpc.snapshot.v2'
SKETCH_PADDED_BYTES = 82_000
FEATURE_PROFILE = {
    'representation': 'ieee754-binary32',
    'dimensions': 512,
    'coordinate_abs_bound': 1,
    'feature_scale_exponent': 149,
    'score_scale_exponent': 298,
    'ring_bits': 320,
}


def validate_publication(snapshot):
    if (snapshot.get('schema') != SNAPSHOT_SCHEMA or
            snapshot.get('publication_policy') != POLICY or
            snapshot.get('public_feature_profile') != FEATURE_PROFILE or
            'provenance' in snapshot):
        raise ValueError('Unsupported public publication policy')
    if snapshot.get('compact_verification') is not None:
        if snapshot.get('private_sketch_plaintext_bytes') != SKETCH_PADDED_BYTES:
            raise ValueError('Unsupported private sketch padding policy')

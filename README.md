# PPCMR

Research code for privacy-preserving cross-modal retrieval with access control and client verification. The current method uses frozen 512-dimensional CLIP FP32 embeddings, exact secret-shared inner products, authenticated scope permissions, and private compact verification.


## Quick start

Python 3.10 or newer is required. The code was checked with Python 3.12 on Windows. From the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_float_demo.py --engine cached
python run_float_demo.py --engine fresh
```

The demo uses synthetic vectors and two separate worker processes. It checks both retrieval directions against an independent exact reference, excludes unauthorized records, and rejects a changed score and an omitted candidate. It does not require CLIP, a GPU, or a dataset.

## Implementation

| File | Purpose |
| --- | --- |
| `float_reference.py` | Lossless FP32-to-integer conversion and certified exact ranking reference |
| `ppcmr/float_mpc.py` | Signed publication, permissions, private query-share envelopes, fresh two-cloud computation |
| `ppcmr/cached_mpc.py` | Snapshot-bound matrix-mask reuse with fresh per-query correlations |
| `ppcmr/compact_verify.py` | Complete catalog coverage, private linear score checks, ranking and content verification |
| `ppcmr/publication_policy.py` | Input-independent public representation parameters |
| `ppcmr/exact_ring.py` | Exact signed-limb arithmetic kernels |
| `ppcmr/streaming_store.py`, `streaming_mpc.py`, `streaming_local.py` | Bounded row-window storage and scalability implementation |
| `ppcmr/local_reference.py`, `cached_local.py` | Authorized local comparison methods |
| `tests/` | Arithmetic, permissions, replay, tampering, leakage-regression and streaming tests |

`ppcmr/protocol.py`, `dataset.py`, and `attacks.py` preserve the earlier synthetic binary-feature reference and its tests. The current floating-point method is demonstrated by `run_float_demo.py`.

Exactness means the exact mathematical inner product of the unchanged FP32 input coordinates, with score descending and record ID ascending for exact ties. FP32 BLAS rounding can produce different ties; `audit_float_reference.py` checks that distinction. The prototype assumes a trusted owner/dealer and non-colluding semi-honest clouds for computation privacy; client verification separately checks arbitrary responses under honest setup and confidential verifier state. Users sharing a scope share its verifier seed/sketch; disclosure by an authorized user can undermine other users' score checks. Users receive the authorized score vector for client-side ranking. The original runners use local IPC; the controlled TCP experiment below adds measured network emulation on one host.

## Real data and CLIP features

Obtain COCO/Flickr30k and their annotations separately under their applicable terms. Obtain the OpenAI CLIP ViT-B/32 weights separately. No dataset or checkpoint is redistributed here.

For COCO, use this local layout:

```text
<coco-root>/
  raw/train2014/...
  raw/val2014/...
  splits/karpathy/dataset_coco.json
```

The loader reads canonical Karpathy metadata directly, keeps the original test image order and first five captions, and assigns ten artificial balanced permission scopes independent of semantics. The default selection seed is `20261007`.

```sh
python prepare_coco.py --data-root /path/to/coco
python -m pip install -r requirements-encoder.txt
python encode_coco.py --folder artifacts/real_coco/test --weights /path/to/open_clip_model.safetensors --device cuda
python audit_float_reference.py --folder artifacts/real_coco/test --output artifacts/accuracy_coco
```

Use `--device cpu` if needed. The encoder loads local weights, uses QuickGELU, disables TF32, and records model/manifest/cache hashes and package versions in the generated `encoder.json`. Optional environment variables `PPCMR_COCO_ROOT` and `PPCMR_CLIP_WEIGHTS` supply local defaults. Library/GPU differences can affect regenerated embeddings; the protocol still checks exact scores for those actual input embeddings.

Flickr30k preparation expects your locally supplied `flickr_annotations_30k.csv` with fields `filename,split,raw,sentids,img_id`, a `flickr30k-images/` image folder, and an alternate `test.txt` list. `raw` and `sentids` are JSON arrays. The CSV's 1000-image test split defines this experiment; the alternate list is audited, not merged.

```sh
python prepare_flickr30k.py --data-root /path/to/flickr30k
python encode_coco.py --folder artifacts/real_flickr30k/test --weights /path/to/open_clip_model.safetensors --device cuda
python audit_float_reference.py --folder artifacts/real_flickr30k/test --output artifacts/accuracy_flickr30k
```

## Research runners

After encoding both datasets, the bounded final-profile runner evaluates COCO at 100/250/1000 images and Flickr30k at 1000 images, plus an epoch change. It uses three distinct query pairs per cohort and excludes one warm-up per direction/method:

```sh
python run_bounded_revision.py --output artifacts/bounded_run
```

The scalability path retains 5000 Karpathy test images and appends 15000 distinct training images sampled with NumPy seed `20261010`. Original embeddings are reused for the first 5k. Galleries beyond 5k are **extended scalability galleries**, not standard test-set recall benchmarks:

```sh
python prepare_scale_gallery.py
python encode_coco.py --folder artifacts/scale_20261010/extension --weights /path/to/open_clip_model.safetensors --device cuda
python prepare_scale_gallery.py --combine
python run_scale_research.py --sizes 1000 --output artifacts/scale_20261010/run_1k
# Optional full scalability run:
python run_scale_research.py --sizes 1000 5000 10000 20000 --output artifacts/scale_20261010/run_full
```

These two runners use Windows working-set counters (`ppcmr/windows_memory.py`). The 20k run can use tens of GB of local scratch storage and generates substantial setup traffic; start with the demo or 1k point. Generated scratch files contain role-private state and must stay local. Complete scores, ordered IDs, and decrypted content are checked independently; setup, online communication and timing are recorded separately. The single trusted harness combines owner, dealer and client state, so its memory is not isolated client memory.

Additional drivers cover full/compact verification, cached/local comparisons and publication/query privacy audits. Existing reports are not bundled. The release removes local machine paths and reconstructs manifests from user-supplied annotations; the cryptographic protocol modules and their tests retain the measured implementation.

## Controlled TCP experiment and current plots

[`experiments/network/README.md`](experiments/network/README.md) documents the bounded 5k/20k-image I2T comparison: five paired queries, one excluded warm-up per method/profile, four calibrated network profiles, fresh and cached MPC, and independent exact-score/order/content checks. Linux network namespaces and receiver-side router egress shaping carry the original streamed protocol over TCP. DU/dealer share one node; the clouds occupy separate processes and namespaces on the same host. This is controlled network emulation, not a cross-cloud VM deployment.

The public runner adapts local paths and records a new pre-run design/source manifest. The numerical protocol modules remain unchanged. The original measurement environment used Linux/WSL2, Python 3.14.8, NumPy 2.4.6, cryptography 47.0.0, OpenBLAS 0.3.30, and four BLAS threads. New runs record their actual versions and generated input hashes; regenerated inputs or other runtimes can change timings.

`plots/plot_accuracy.py` and `plots/plot_scale_costs.py` regenerate the current figure layouts from locally supplied measurement CSVs. Install `requirements-plots.txt` separately. The accuracy plot shows Exact curves; visually overlapping FP32 curves are omitted explicitly, with ranking/recall differences reported by the arithmetic audit. Both plots use four visible spines, grids on both axes, and inward ticks on all sides. Source CSVs and rendered figures are not bundled.

```sh
python -m pip install -r requirements-plots.txt
python plots/plot_accuracy.py --data-dir /path/to/local/csvs --output-dir artifacts/plots
python plots/plot_scale_costs.py --data-dir /path/to/local/csvs --output-dir artifacts/plots
python plots/summarize_arithmetic.py --coco-audit artifacts/accuracy_coco/arithmetic_audit.json --flickr-audit artifacts/accuracy_flickr30k/arithmetic_audit.json --output artifacts/plots/arithmetic_differences.csv
```

## 中文说明

仓库仅包含本方案代码、测试、依赖与运行说明。先运行 `python -m unittest discover -s tests -v` 和 `python run_float_demo.py`，随后再按上述步骤配置本地数据。正式方法使用 CLIP 浮点特征；早期二值原型仅为历史测试保留。


# Bounded controlled TCP experiment

This code reproduces the experiment design using local data. It contains no
dataset, embeddings, recorded results, bootstrap pickles, or verifier state.
`tcp_engine.py` reuses `StreamingEngine` and `_stream_worker`; only transport and
process bootstrap change. The exact arithmetic, authentication, compact
verification, block size (256), and request schedule are retained.

## Portable transport check

From the repository root, with the normal requirements installed:

```sh
python experiments/network/test_tcp_transport.py
python -m unittest discover -s tests -v
```

The framing check covers empty, small, and 6 MiB messages and validates counters.
It is a loopback transport check, not a latency benchmark.

## Linux measurement environment

Use a dedicated Linux host/VM or WSL2 instance with Python, the repository's
requirements, `iproute2` (including `tc`/netem), `ethtool`, `iperf3`, and `ping`.
Root or equivalent network capabilities are needed. Use an ordinary filesystem
with sufficient private scratch space (the full run can use tens of GB).
The scripts create four namespaces named `ppcmrn16-{h,c0,c1,r}` and refuse to
reuse an existing one. Create succeeds before calibration or execution; clean
up only the namespaces created for this run. There is no host-wide network reset.
Socket buffer maxima are raised within this dedicated experiment kernel.

Prepare the extended COCO gallery as described in the main README. Set the
paths below to absolute Linux paths. `--coco-root` points to the dataset root
containing the original manifest-relative image paths; it overrides the root
recorded by a Windows-generated manifest without changing record identities or
hashes. Do not edit the input manifest or encoder provenance.

```sh
export PYTHONPATH="$PWD"
NET="$PWD/experiments/network"
GALLERY="$PWD/artifacts/scale_20261010/gallery"
COCO_ROOT=/path/to/coco
RUN="$PWD/artifacts/network_run"
SCRATCH=/path/to/new/private-network-scratch

sudo -E python "$NET/network_topology.py" create
sudo -E python "$NET/calibrate_network.py" --output "$RUN/calibration"
# Start with this small transport/protocol run using a distinct output/scratch:
sudo -E ip netns exec ppcmrn16-h python "$NET/run_network_experiment.py" \
  --gallery "$GALLERY" --coco-root "$COCO_ROOT" \
  --output "$RUN/smoke" --scratch "$SCRATCH-smoke" --smoke
# Full bounded design (can take several hours):
sudo -E ip netns exec ppcmrn16-h python "$NET/run_network_experiment.py" \
  --gallery "$GALLERY" --coco-root "$COCO_ROOT" \
  --output "$RUN/executed" --scratch "$SCRATCH"
python "$NET/summarize_network.py" --input "$RUN/executed/experiment.json" \
  --gallery "$GALLERY" --calibration "$RUN/calibration" --output "$RUN/results"
sudo -E python "$NET/network_topology.py" cleanup
```

Use the same Python environment under `sudo` (an absolute interpreter path if
needed). After a failure, preserve partial evidence and clean up the namespaces
created by the successful `create` call. Use new output and scratch paths for a
new run. Private worker pickles are mode 0600 and excluded from version control.
The bootstrap uses a trusted shared local filesystem, matching the measured
harness; it is outside the measured protocol. Protocol matrices, correlations,
openings, candidate responses, score responses and content fetches use TCP.

## Design and reporting

DU and dealer share H; CS0 and CS1 use separate processes/namespaces. Three
independent full-duplex links connect each node pair. Shaping is on router
egress, downstream of the sender; virtual aggregation is disabled. Profiles:

| Profile | Target RTT | Per-direction link rate |
| --- | --- | --- |
| N0 | Native | Uncapped |
| N1 | 20 ms | 1 Gbit/s |
| N2 | 80 ms | 1 Gbit/s |
| N3 | 80 ms | 100 Mbit/s |

Calibration records actual RTT and bidirectional throughput for all three links.
The full design uses 5k/20k images (25k/100k captions), I2T, k=10, all scopes,
five query indices selected from the first 5k images with seed 20261016, and one
excluded warm-up per method/profile. This gives 80 formal and 16 warm-up queries.
Every complete score vector, deterministic ordered IDs and original selected
contents are checked against an independent exact reference. Method order is
alternated by profile and gallery. A 24-hour owner-signed credential prevents
expiry during the long sequential run; validation is unchanged.

The runner writes `design.json` before publication/setup and binds its digest
into the incrementally saved `experiment.json`. The auditor requires completion,
checks source/input hashes against that plan, all query counts and correctness
flags, unique request digests, cross-profile score/order consistency, and qdisc
drop deltas. It refuses a partial or smoke cohort as the formal experiment.
The public path/provenance wrapper writes its own plan; it does not claim the
byte hashes of the archived original runner.

Complete query timing includes query construction, fresh dealer work/delivery,
server scoring, catalog/score transport, client ranking, selected content fetch,
and verification. Publication/bootstrap/matrix setup are separate. Components
can overlap and must not be added again to the total. Summaries use medians,
minima/maxima and paired speedups for these five queries; they do not establish
p95 or performance across independent physical clouds. Cached local search is
not measured in this TCP cohort.

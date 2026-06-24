# OpenSemCom

OpenSemCom is a runnable research prototype for risk-certified open-environment semantic communication in dynamic wireless settings. It implements the technical spine from `OpenSemCom_Research_Plan.md`:

```text
open semantic risk -> selective reliability -> safe adaptation -> semantic outage -> wireless resource control
```

The default code path is intentionally real-data-only: every experiment must be pointed at an explicit dataset manifest. There is no no-data fallback.

## What Is Included

- AWGN, Rayleigh, Rician, MIMO, Doppler, blockage, burst-noise, and interference channel models.
- Layered semantic parser and encoder with core, refinement, evidence, and fallback payloads.
- Selective semantic receiver with accept, refine, semantic-HARQ, adapt, reject/open, and fallback decisions.
- Channel-task-aware open-risk detector.
- Conformal calibration and prediction sets.
- Safe adaptation gate with finite-sample Hoeffding margin.
- Reliability-card codec library and codec router.
- Risk-aware scheduler with power, bandwidth, latency, energy, and compute constraints.
- OpenSemCom-Bench manifest regimes for closed-ID, channel-open, source-open, class-open, task-open, supervision-limited, resource-open, and full-open evaluation.
- Metrics for open semantic risk, outage, calibration, open-set detection, semantic goodput, and adaptation safety.

## Quick Start

```powershell
python -m pip install -e .[dev]
opensemcom-demo --dataset-manifest path/to/manifest.csv
opensemcom-bench --dataset-manifest path/to/manifest.csv --regime full-open --samples 512 --users 4 --seed 7
pytest
```

The demo prints a compact JSON summary with the main research metrics:

- `open_semantic_risk`
- `semantic_outage`
- `open_semantic_outage`
- `coverage`
- `semantic_goodput`
- `adaptation_harm_rate`
- `risk_per_joule`
- `risk_latency_product`

## Repository Layout

```text
src/opensemcom/
  adaptation.py       safe adapter and candidate update gate
  benchmark.py        OpenSemCom-Bench regimes and runner
  calibration.py      split conformal calibration
  channels.py         wireless channel models
  codec.py            reliability-card codec library
  config.py           typed configuration
  harq.py             semantic HARQ/refinement loop
  metrics.py          reliability, calibration, and open-world metrics
  receiver.py         selective semantic receiver
  risk.py             open semantic risk and detector
  scheduler.py        risk-aware resource scheduler
  semantic.py         parser, semantic layers, encoder, decoder
  simulation.py       end-to-end system loop
  types.py            shared typed records and decisions
```

## Manifest Format

Experiments require a CSV manifest with these columns:

```csv
source_path,label,task,domain,is_unknown,split,regime
```

`source_path` must point to a real local artifact. Supported artifact formats are `.npy`, `.npz`, or any non-empty file whose bytes can be converted into a fixed-size baseline feature vector. The manifest can use `split=calibration`, `split=train`, or `split=eval`, and `regime` can select one of the OpenSemCom regimes.

## Design Notes

This is a research scaffold, not a claim of final empirical performance. The manifest loader is deliberately strict so experiments cannot accidentally run without real data. Image, sensing, and text datasets should be integrated through the same `SemanticSample` and `BenchmarkRegime` abstractions.

# OpenSemCom Paper Validation Plan

This plan turns the unified certificate contract into falsifiable paper
claims. A result enters the main paper only when its policy has a valid
independent certificate and its artifacts pass the checks below.

## Central claim

At a predeclared accepted-outage target and resource budget, OpenSemCom's
fixed progressive policy delivers more correct accepted semantic decisions
than strong alternatives while retaining a valid finite-sample certificate.

This claim has three parts:

1. **Reliability:** the independent upper bound is at or below the target.
2. **Utility:** semantic goodput is higher than certified comparison methods.
3. **Efficiency:** the gain is not explained by unconstrained resource use.

## Main contributions to test

### C1. Composed-policy reliability certificate

Test the complete accept/refine/HARQ/reject route, not isolated stages.
Report certificate size, accepted count, unsafe count, exact upper bound,
confidence, target, and certificate validity.

Required evidence:

- nominal closed-ID sanity check;
- full-open mild, medium, hard, and extreme cohorts;
- targets 1%, 2%, 5%, and 10%;
- no threshold or route selection on the certificate cohort;
- reject-all deployment when certification fails.

### C2. Certified progressive semantic control

Compare the progressive policy against:

- receiver-only selection;
- no-channel-metadata receiver;
- DINO-only detector;
- foundation-feature ensemble detector;
- fixed refinement;
- communication baselines using official checkpoints where available.

All methods receive their own policy-selection cohort and are evaluated on
the same untouched certificate and evaluation cohorts. Compare goodput only
between methods whose certificate is valid at the target.

### C3. Safe behavior under channel shift

Use two separate protocols:

1. **Held-out shift:** calibrate on a nominal channel and evaluate on a
   different family or out-of-envelope SNR. Success means unsafe immediate
   acceptance is prevented; useful goodput may fall to zero.
2. **Deployment-conditioned operation:** calibrate and certify on an i.i.d.
   cohort from the declared channel regime. Success means useful
   certified goodput, not merely rejection.

Minimum channel set:

- AWGN;
- Rayleigh;
- Rician;
- MIMO;
- interference/CSI error;
- one realistic 5G NR TDL or CDL model when implemented.

Nakagami and additional 5G models are extensions, not prerequisites for the
first certificate paper.

### C4. Sequential adaptation as a supporting result

Keep this out of the headline unless verified feedback is available. Compare:

- frozen receiver;
- naive same-buffer adaptation;
- independent-split sequential gate;
- gated adaptation followed by recertification.

The paper must report candidate acceptance rate, realized harm rate,
alpha-spending sequence, recertification delay, and post-recertification
goodput. Without post-update recertification, adapted predictions cannot be
accepted by the certificate-enforced policy.

## Statistical protocol

- Predeclare target, confidence, severity mixture, resource budget, methods,
  seeds, and primary metric before inspecting certificate or evaluation
  results.
- Predeclare one primary policy, or set `--certificate-family-size` to the
  number of policies eligible for post-certificate selection. The latter uses
  Bonferroni allocation and reports both family and per-policy alpha.
- The confirmatory primary policy is `opensemcom_progressive` at 5% accepted
  outage and resource budget 0.80. It is selected only on the policy-selection
  cohort and is never replaced after certificate inspection.
- Use source-disjoint model-fit, policy-selection, certificate, and evaluation
  cohorts. Preserve their SHA-256 hashes.
- For generic conformal experiments, retain a separate conformal cohort as
  specified in `CERTIFIED_FRAMEWORK.md`.
- Use at least five training/split seeds.
- Report seed-level values and confidence intervals; do not report only the
  best seed.
- Use paired evaluation cohorts for method comparisons.
- Treat target outage as a feasibility constraint, never as a score.
- Do not tune methods after reading certificate failures. A changed policy
  requires a fresh untouched certificate cohort.

## Required sample planning

At 95% confidence and zero unsafe accepted certificate examples:

| Outage target | Minimum accepted certificate examples |
|---:|---:|
| 1% | 299 |
| 2% | 149 |
| 5% | 59 |
| 10% | 29 |

For the secondary simultaneous eight-method sensitivity analysis, Bonferroni
allocation uses \(\alpha_j=0.05/8=0.00625\). At a 5% outage target this raises
the zero-error minimum from 59 to 99 accepted certificate examples.

For a severity with expected safe coverage \(c\), allocate enough total rows
that the independent certificate cohort is expected to contain substantially
more than the minimum. Use a margin of at least 25%; planning for exactly 59
accepted examples at a 5% target is fragile.

The extreme 91% open mixture is the binding case. With a certificate fraction
of 70%, the canonical 192-known/2,000-open policy/certificate candidate pools
use all 192 known candidates after matching the 91% mixture and yield about
134 known certificate rows. This is above both the pointwise minimum of 59
and the secondary eight-method familywise minimum of 99 at a 5% target, but
still requires a highly selective, nearly error-free policy. Increase known
feature coverage before treating the extreme condition as a stable result.

## Artifact acceptance checklist

A main-table run must include:

- source manifests and validation output;
- model-fit, policy-selection, certificate, and evaluation hashes;
- model checkpoints and environment versions;
- seed-level policy and summary CSV files;
- exact certificate fields;
- evaluation traces or sufficient accepted/unsafe indices for audit;
- Slurm logs and exit status;
- proof that no source identity crosses cohort boundaries.

Reject a run from the main table when:

- its certificate is invalid;
- its certificate distribution does not match the stated claim;
- fewer than the required accepted certificate examples are available;
- a model, route, or threshold changed after certification;
- a baseline uses a weaker data split or different safety definition;
- a claimed physical metric is only a cost proxy.

## Immediate run order

1. Rerun the focused communication-control suite with the new independent
   certificate protocol using
   `slurm/run_certified_communication_control.slurm`.
2. Inspect certificate feasibility and increase calibration coverage where
   needed, especially for extreme full-open exposure.
3. Complete all 2,411 DeepSense feature rows and rerun independently certified
   exact-beam top-k experiments.
4. Run certified ablations.
5. Add realistic 5G channel models and run held-out versus
   deployment-conditioned channel protocols.
6. Run adaptation only after a verified-feedback and recertification workflow
   is available.

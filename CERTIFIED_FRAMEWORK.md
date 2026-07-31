# OpenSemCom Certificate Contract

This document is the normative statement of the guarantees implemented by
OpenSemCom. The broader research plan contains hypotheses and future
directions; only claims stated here and exercised by tests should be described
as implemented guarantees.

## 1. Unified decision problem

For one deployment example, let

\[
Z=(X,Y,H,\tau,D,U)
\]

contain the source, target, channel state, task, domain, and open-exposure
indicator. A fixed OpenSemCom policy \(\pi\) observes everything available at
deployment except \(Y\) and emits a final prediction \(\widehat Y\) and action

\[
A_\pi(Z)\in\{\mathrm{accept},\mathrm{reject}\}.
\]

The action includes the complete route through the progressive protocol:
initial transmission, zero or more semantic-refinement/HARQ rounds, and the
final accept or reject decision.

The binary unsafe loss used by the current experiments is

\[
L_\pi(Z)
=
\mathbf 1\{
\widehat Y\ne Y
\;\lor\;
U=1
\},
\]

where \(U=1\) means unknown class, unseen task, or unseen domain according to
the manifest and configured training scope.

The accepted-outage risk is

\[
R_{\mathrm{acc}}(\pi)
=
\Pr\!\left(L_\pi(Z)=1\mid A_\pi(Z)=\mathrm{accept}\right).
\]

The primary optimization metric is semantic goodput

\[
G(\pi)
=
\mathbb E\!\left[
\mathbf 1\{A_\pi(Z)=\mathrm{accept}\}
\mathbf 1\{\widehat Y=Y\}
\right],
\]

subject to

\[
R_{\mathrm{acc}}(\pi)\le\delta
\]

and the configured resource constraints. In code, goodput and decision
coverage are divided by the number of evaluated examples. Efficiency per
channel use is reported separately.

## 2. Four disjoint data roles

Certification requires four disjoint subsets:

1. **Model-fit split:** fits receiver prototypes or heads, the open-risk
   detector, and the empirical channel-support profile.
2. **Conformal split:** fits prediction-set nonconformity thresholds after the
   model is fixed.
3. **Policy-selection split:** chooses accept thresholds and refinement
   thresholds.
4. **Certificate split:** is inspected exactly once after the complete
   multi-stage policy is fixed.

The default fractions are 50%, 20%, 15%, and 15%. Runs with too few examples,
missing mixed-open calibration for a semantic-open regime, or too few accepted
certificate examples receive an invalid certificate and cannot emit an
accepted decision while certificate enforcement is enabled.

This separation is essential. Selecting a threshold and certifying it on the
same examples does not provide the implemented guarantee.

The focused `communication_control_suite` does not produce conformal
prediction sets. It therefore uses model-fit, policy-selection, certificate,
and evaluation cohorts; its end-to-end certificate remains covered by
Theorem 1, while no conformal-set claim is made for that suite.

## 3. Finite-sample accepted-outage certificate

Let the fixed policy be evaluated on \(n\) independent certificate examples.
Let

\[
M=\sum_{i=1}^{n}\mathbf 1\{A_\pi(Z_i)=\mathrm{accept}\}
\]

and

\[
K=\sum_{i=1}^{n}
\mathbf 1\{A_\pi(Z_i)=\mathrm{accept}\}L_\pi(Z_i).
\]

OpenSemCom computes the one-sided Clopper-Pearson upper confidence limit

\[
U_{\mathrm{CP}}(K,M;\alpha).
\]

The certificate is valid only when

\[
M\ge M_{\min}
\quad\text{and}\quad
U_{\mathrm{CP}}(K,M;\alpha)\le\delta.
\]

### Theorem 1: fixed-policy selective reliability

Assume:

1. the certificate examples, including channel realizations and any policy
   randomization, are i.i.d. draws from the deployment distribution;
2. the complete policy \(\pi\), including every refinement threshold, channel
   support rule, scheduler rule, and model parameter, is fixed before the
   certificate split is inspected;
3. unsafe acceptance is the bounded binary loss defined above.

Then

\[
\Pr\!\left(
R_{\mathrm{acc}}(\pi)
\le
U_{\mathrm{CP}}(K,M;\alpha)
\right)
\ge 1-\alpha.
\]

Consequently, if OpenSemCom emits a valid certificate with upper bound at most
\(\delta\), then

\[
\Pr\!\left(R_{\mathrm{acc}}(\pi)\le\delta\right)\ge1-\alpha.
\]

### Proof

Condition on \(M=m>0\). Because the policy is fixed independently of the
certificate split and the examples are i.i.d., the accepted examples are
i.i.d. draws from the deployment distribution conditional on acceptance.
Their unsafe indicators are Bernoulli variables with parameter
\(R_{\mathrm{acc}}(\pi)\), and
\(K\mid M=m\) is binomial with parameters \(m\) and
\(R_{\mathrm{acc}}(\pi)\). By construction, the one-sided
Clopper-Pearson interval covers the binomial parameter with probability at
least \(1-\alpha\). The claim holds for every \(m>0\), hence also after
averaging over the random value of \(M\). When \(M=0\), OpenSemCom declares the
certificate unavailable rather than claiming conditional reliability. \(\square\)

### Multiple policies

Theorem 1 is a pointwise guarantee for one policy fixed before certification.
If a paper will select among \(J\) certified policies after inspecting their
certificates, OpenSemCom uses Bonferroni allocation
\(\alpha_j=\alpha_{\mathrm{family}}/J\). A union bound then gives simultaneous
coverage of every policy in the declared family with probability at least
\(1-\alpha_{\mathrm{family}}\). The focused suite exposes this through
`--certificate-family-size`. Leaving that value at one permits only a
predeclared per-policy claim, not a post-certificate familywise selection
claim.

### Why the composed policy is certified

OpenSemCom does not combine independent stage-wise claims. It first fixes all
stage policies on the policy-selection split, runs the complete
accept/refine/reject trajectory on each certificate example, and certifies the
final outcomes. Therefore, adaptive routing through HARQ is already part of
the fixed policy covered by Theorem 1.

## 4. Conformal prediction sets

The receiver uses split-conformal probability nonconformity

\[
s(x,y)=1-p_y(x).
\]

The receiver model is fixed before the conformal split is used. Under
exchangeability between conformal and deployment examples, the usual
split-conformal marginal prediction-set coverage applies. This prediction-set
result is not treated as an accepted-outage certificate by itself. The final
Clopper-Pearson certificate evaluates the full decision policy, including any
imperfection in the conformal set.

## 5. Channel support and shift

The model-fit split creates an empirical envelope over channel family, SNR,
effective SNR, gain, interference, blockage, Doppler, CSI error, and available
PHY diagnostics. A changed channel family has a distinct channel-family code
and is outside the envelope.

An out-of-envelope example cannot be immediately accepted. It may request
refinement; if support is not recovered before the refinement budget is
exhausted, it is rejected.

This is a conservative domain-of-use guard, not a universal generalization
theorem. A certificate fitted on one channel distribution does not cover an
arbitrary shifted distribution merely because a point lies numerically inside
the empirical envelope. Claims under an in-support distribution shift require
one of:

- i.i.d. calibration and certificate examples from that deployment distribution;
- a predeclared group-conditional certificate with i.i.d. examples per group;
- a valid covariate-shift weighting argument with bounded density ratios; or
- a distributionally robust bound under an explicitly specified uncertainty
  set.

Without one of these conditions, OpenSemCom must report the shift result as
empirical rather than certified.

## 6. Sequential adaptation

Adaptation is disabled by default. When explicitly enabled, it requires:

- verified labels and corresponding received representations;
- one proposal split used to construct the candidate;
- a separate validation split used only by the gate.

At update \(t\), define the paired bounded loss difference

\[
D_i
=
\ell(\psi'_t;Z_i)-\ell(\psi_t;Z_i)
\in[-1,1].
\]

Use the confidence allocation

\[
\alpha_t=\frac{\alpha}{t(t+1)},
\qquad
\sum_{t=1}^{\infty}\alpha_t=\alpha,
\]

and radius

\[
\epsilon_t
=
\sqrt{\frac{2\log(1/\alpha_t)}{|V_t|}}.
\]

The candidate is accepted only if

\[
\overline D_t+\epsilon_t\le-\kappa.
\]

### Theorem 2: sequential non-degradation gate

Assume each candidate is fixed before its fresh independent verified
validation split is inspected, validation examples are i.i.d. draws from the
update's target distribution, no validation example is reused at a later
update, and losses lie in \([0,1]\). Then, with probability at
least \(1-\alpha\), every accepted update satisfies

\[
\mathbb E[D_t]\le-\kappa.
\]

### Proof

For one update, \(D_i\in[-1,1]\). Hoeffding's inequality gives

\[
\Pr\!\left(
\mathbb E[D_t]>
\overline D_t+
\sqrt{\frac{2\log(1/\alpha_t)}{|V_t|}}
\right)
\le\alpha_t.
\]

If the gate accepts, its right-hand side is at most \(-\kappa\). A union bound
over all updates gives total failure probability at most
\(\sum_t\alpha_t=\alpha\). \(\square\)

The currently gated loss is receiver classification error, not conditional
accepted outage. Therefore, after any accepted model update, OpenSemCom
invalidates the old decision certificate and prohibits acceptance until the
adapted composed policy is recertified. This preserves the distinction between
safe model updating and certified selective operation.

## 7. Scheduler claim

The scheduler enumerates a finite set of layer, repetition, power, bandwidth,
latency, energy, compute, and codec actions. It returns the feasible action
with the smallest configured surrogate score. This is an exact minimizer over
the enumerated surrogate candidates.

It is not currently a proof of optimality for true physical latency, energy,
or semantic risk. Those quantities require measured cost and response models.

## 8. Certificate sample sizes

At 95% confidence and zero observed unsafe acceptances, the minimum accepted
certificate examples are:

| Target outage | Minimum accepted certificate examples |
|---:|---:|
| 1% | 299 |
| 2% | 149 |
| 5% | 59 |
| 10% | 29 |

These are accepted examples, not total calibration rows. If the independent
certificate split is 15% of calibration data and expected certificate
coverage is \(c\), a rough zero-error planning lower bound is

\[
N_{\mathrm{cal}}
\ge
\frac{M_{\min}}{0.15c}.
\]

More data are required when unsafe acceptances occur, when certifying multiple
predeclared groups, or when using lower-coverage policies.

## 9. Implemented non-claims

The current implementation does not claim:

- conditional coverage for every channel state or semantic subgroup;
- safety under arbitrary, undetected distribution shift;
- adaptation without verified feedback;
- monotonic improvement from every refinement round;
- physical energy or wall-clock latency optimality;
- a certificate for archived runs that predate the four-way split.

These are experiment or future-theory targets unless a later version adds the
necessary assumptions, algorithms, and tests.

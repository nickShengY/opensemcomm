# OpenSemCom: Risk-Certified Open-Environment Semantic Communication for Dynamic 6G Wireless Networks

## Executive Summary

**OpenSemCom** is a risk-certified semantic communication framework for dynamic 6G wireless networks. It is designed for deployment environments where the source distribution, wireless channel, semantic label space, task objective, supervision level, and edge-resource budget may change after training.

The central thesis is:

> A semantic communication system should not merely decode meaning. It should also estimate whether the decoded meaning is reliable enough to use, decide whether to accept, refine, retransmit, adapt, or reject/open, and control wireless resources according to calibrated semantic risk.

OpenSemCom is built around one unified objective:

\[
\min_{\pi} \mathcal{R}_{\mathrm{open}}(\pi)
\]

subject to wireless constraints on power, bandwidth, latency, energy, compute, and a semantic reliability constraint:

\[
\mathbb{P}\!\left(
\ell_\tau(\hat{Y}_t,Y_t)>\epsilon_\tau
\mid d_t=\mathrm{accept}
\right)
\le \delta .
\]

The research contribution is a mathematically grounded framework for **open-environment semantic reliability**, combining:

- open semantic risk;
- open semantic outage;
- selective semantic decoding;
- conformal reliability calibration;
- semantic HARQ/refinement;
- risk-certified safe adaptation;
- reliability-card semantic codec routing;
- wireless resource-aware semantic scheduling;
- benchmark design for simultaneous source, channel, class, task, supervision, and resource shifts.

---

## 1. Research Vision

Semantic communication aims to transmit task-relevant meaning rather than raw bits. This paradigm is attractive for 6G wireless systems because many emerging applications care more about **task success** than exact signal reconstruction. Examples include autonomous driving, edge AI inference, digital twins, smart city sensing, vehicular networks, robotic control, extended reality, and mission-critical IoT.

However, semantic communication also creates a new type of failure. A conventional communication system fails when bits are corrupted. A semantic communication system may fail more subtly: it may output a plausible but wrong semantic interpretation with high confidence. This is especially dangerous when the deployment environment differs from training.

OpenSemCom targets the following deployment setting:

\[
\text{source shift}
+
\text{channel shift}
+
\text{open semantic classes}
+
\text{task drift}
+
\text{limited supervision}
+
\text{edge-resource constraints}.
\]

The goal is not only to improve average semantic accuracy, but to make semantic communication **reliable, selective, adaptive, and resource-aware** under open 6G wireless environments.

---

## 2. Literature Landscape and Research Gap

### 2.1 Learned JSCC and Semantic Communication

Deep learning based joint source-channel coding has shown that learned encoders and decoders can directly map source data to channel symbols and avoid the cliff effect observed in separated digital systems. [DeepJSCC][1] is a foundational example for wireless image transmission. DeepSC introduced a Transformer-based text semantic communication system that focuses on recovering sentence meaning rather than exact bits. [DeepSC][2] WITT further advanced image semantic communication using a Swin Transformer backbone and channel-state-information-aware latent modulation. [WITT][3]

These methods demonstrate the potential of semantic communication, but they are usually evaluated in settings where the task, semantic space, and deployment conditions are relatively controlled.

### 2.2 Task-Oriented and Task-Agnostic Semantic Communication

Task-oriented semantic communication optimizes communication for downstream inference rather than reconstruction. More recent task-agnostic semantic communication methods aim to support multiple tasks or modalities using information bottleneck principles, multimodal representations, or foundation-model features. [TASC][4] [SemCLIP][5]

These methods reduce task specificity, but they typically do not fully address the case where deployment involves **unknown semantic classes, unseen task objectives, unsafe adaptation, and wireless resource decisions simultaneously**.

### 2.3 Robust and Distributionally Robust Semantic Communication

Robust semantic communication has begun to address uncertainty in source and channel conditions. WaSeCom, for example, uses Wasserstein distributionally robust optimization to improve semantic robustness against semantic-level and transmission-level uncertainties. [WaSeCom][6]

This direction is highly relevant, but open-environment semantic communication requires more than robustness to perturbations. It also requires **selective acceptance, open-set rejection, calibrated risk, safe adaptation, and resource control**.

### 2.4 Conformal Reliability for Semantic Communication

Conformal semantic communication has recently introduced distribution-free task-level reliability guarantees under changing wireless channel conditions. [ConformalSemCom][7] This is an important step toward reliable semantic communication.

OpenSemCom extends this reliability direction from channel shift alone toward a multi-axis open environment involving source shift, task shift, class novelty, adaptation risk, and wireless resource constraints.

### 2.5 OOD Detection, Open-Set Recognition, and Test-Time Adaptation

OOD detection and open-set recognition study whether a test input comes from a known or unknown distribution. Recent analysis emphasizes that semantic shift and covariate shift should be separated carefully. [OODOSR][8] Test-time adaptation aims to adapt models during deployment, but surveys and benchmarks show that online adaptation can be unstable and sensitive to pseudo-label noise, inconsistent settings, and accumulated errors. [OTTA][9] [TTAB][10]

OpenSemCom imports these ideas into a wireless semantic communication loop, where semantic uncertainty, channel uncertainty, and resource allocation interact.

### 2.6 6G Motivation

The IMT-2030/6G vision emphasizes ubiquitous intelligence, resilience, sustainability, and advanced connectivity. [ITU6G][11] Semantic communication fits this direction, but practical 6G systems require reliability under dynamic channels, mobility, interference, blockage, edge computation limits, and changing sensing environments.

---

## 3. Core Research Gaps

OpenSemCom targets seven concrete research gaps.

| Gap | Limitation in Current Work | OpenSemCom Response |
|---|---|---|
| **G1: Closed semantic world** | Many systems assume known classes, known tasks, and known source domains. | Formalize open-environment semantic communication with source, channel, class, task, supervision, and resource shifts. |
| **G2: Channel robustness is not enough** | Robustness to SNR or fading does not guarantee semantic reliability under unseen meanings or tasks. | Jointly model semantic openness and wireless channel openness. |
| **G3: No unified open semantic risk** | BER, PSNR, SSIM, LPIPS, and average task accuracy miss unknown-class acceptance and unsafe adaptation. | Define open semantic risk combining task error, unknown acceptance, task mismatch, adaptation harm, calibration error, and resource cost. |
| **G4: Forced semantic prediction** | Existing receivers often output a prediction even when unreliable. | Design a selective receiver that can accept, refine, retransmit, reject/open, or adapt safely. |
| **G5: Unsafe test-time adaptation** | Pseudo-label adaptation can accumulate errors and degrade reliability. | Introduce a certified adaptation gate with high-probability non-degradation guarantees. |
| **G6: Resource control lacks semantic risk certification** | Schedulers often optimize quality or accuracy, not calibrated open risk. | Optimize power, bandwidth, semantic rate, latency, and compute under semantic outage constraints. |
| **G7: Lack of full-open benchmarks** | Benchmarks often vary only SNR, channel type, or source domain. | Build OpenSemCom-Bench with factorial source, channel, class, task, supervision, and resource shifts. |

---

## 4. Problem Statement

### 4.1 Core Research Question

How can a semantic communication system remain reliable when the wireless channel changes, the source distribution shifts, new semantic classes or tasks emerge, supervision is limited, and edge devices operate under strict power, bandwidth, latency, energy, and compute constraints?

### 4.2 Technical Objective

Design a semantic communication policy \(\pi\) that minimizes open semantic risk:

\[
\min_{\pi} \mathcal{R}_{\mathrm{open}}(\pi)
\]

subject to:

\[
\mathbb{E}[P_t] \le P_{\max},
\]

\[
\mathbb{E}[B_t] \le B_{\max},
\]

\[
\mathbb{E}[L_t] \le L_{\max},
\]

\[
\mathbb{E}[E_t] \le E_{\max},
\]

\[
\mathbb{E}[C_t] \le C_{\max},
\]

and:

\[
\mathbb{P}\!\left(
\ell_\tau(\hat{Y}_t,Y_t)>\epsilon_\tau
\mid d_t=\mathrm{accept}
\right)
\le \delta.
\]

Here:

- \(P_t\): transmit power;
- \(B_t\): allocated bandwidth;
- \(L_t\): latency;
- \(E_t\): energy consumption;
- \(C_t\): receiver computation cost;
- \(d_t\): receiver decision;
- \(\epsilon_\tau\): task-specific semantic error tolerance;
- \(\delta\): allowed semantic reliability violation probability.

---

## 5. System Model

### 5.1 Multi-User Wireless Semantic Communication

Consider \(K\) wireless users communicating semantic information to an edge receiver or base station. At time \(t\), user \(k\) observes source data:

\[
X_{k,t} \sim P_t(X),
\]

with latent semantic state:

\[
S_{k,t},
\]

and task target:

\[
Y_{k,t}^{\tau} = g_{\tau_t}(S_{k,t}).
\]

The task is:

\[
\tau_t \in \mathcal{T}_t.
\]

The wireless channel is:

\[
\mathbf{H}_{k,t}.
\]

A multi-user received signal can be written as:

\[
\mathbf{r}_{k,t}
=
\mathbf{H}_{k,t}\mathbf{x}_{k,t}
+
\sum_{j\ne k}
\mathbf{H}_{j,k,t}\mathbf{x}_{j,t}
+
\mathbf{n}_{k,t}.
\]

The transmitted signal is generated by a semantic encoder:

\[
\mathbf{x}_{k,t}
=
f_{\theta,m_{k,t}}
\left(
X_{k,t},
\tau_t,
a_{k,t}
\right),
\]

where:

- \(f_{\theta,m}\): semantic encoder module \(m\);
- \(a_{k,t}\): scheduling action;
- \(m_{k,t}\): selected codec module.

The receiver outputs:

\[
\hat{Y}_{k,t}
=
q_{\psi_t,m_{k,t}}
\left(
\mathbf{r}_{k,t},
\tau_t
\right),
\]

and a reliability-control decision:

\[
d_{k,t}
\in
\{
\mathrm{accept},
\mathrm{refine},
\mathrm{semantic\text{-}HARQ},
\mathrm{adapt},
\mathrm{reject/open}
\}.
\]

This decision set is central. A semantic receiver should not always output a final answer. It should decide whether the decoded meaning is sufficiently reliable.

---

## 6. Open-Environment Definition

The training environment is:

\[
e_0
=
\left(
P_0(X,S,Y),
P_0(\mathbf{H}),
\mathcal{Y}_0,
\mathcal{T}_0,
\mathcal{B}_0
\right).
\]

The deployment environment at time \(t\) is:

\[
e_t
=
\left(
P_t(X,S,Y),
P_t(\mathbf{H}),
\mathcal{Y}_t,
\mathcal{T}_t,
\mathcal{B}_t
\right).
\]

An open semantic environment occurs when at least one of the following holds:

\[
P_t(X,S,Y) \ne P_0(X,S,Y),
\]

\[
P_t(\mathbf{H}) \ne P_0(\mathbf{H}),
\]

\[
\mathcal{Y}_t \not\subseteq \mathcal{Y}_0,
\]

\[
\mathcal{T}_t \not\subseteq \mathcal{T}_0,
\]

\[
\mathcal{B}_t \ne \mathcal{B}_0.
\]

A full-open deployment occurs when multiple shifts happen simultaneously:

\[
e_t
\neq e_0
\quad
\text{through source, channel, class, task, supervision, and resource changes.}
\]

---

## 7. Open Semantic Risk

### 7.1 Task Loss

Let the task loss be:

\[
\ell_\tau(\hat{Y},Y) \in [0,1].
\]

This may represent classification error, normalized detection loss, segmentation error, retrieval loss, captioning loss, or control-relevant semantic error.

### 7.2 Unknown-Class Acceptance Loss

\[
\ell_{\mathrm{unk}}
=
\mathbf{1}
\left[
Y \notin \mathcal{Y}_0
\ \land\
d=\mathrm{accept}
\right].
\]

This penalizes accepting an unknown semantic class as if it were known.

### 7.3 Task-Mismatch Loss

\[
\ell_{\mathrm{task}}
=
\mathbf{1}
\left[
\tau_t \notin \mathcal{T}_0
\ \land\
d=\mathrm{accept}
\ \land\
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\right].
\]

This penalizes unreliable acceptance under task drift.

### 7.4 Adaptation Harm

\[
\ell_{\mathrm{adapt}}
=
\max
\left(
0,
R_t(\psi_t)-R_t(\psi_{t-1})
\right).
\]

This measures degradation caused by adaptation.

### 7.5 Calibration Error

For prediction set \(\Gamma(X,H,\tau)\), define:

\[
\ell_{\mathrm{cal}}
=
\left|
\mathbb{P}\!\left(
Y \in \Gamma(X,H,\tau)
\right)
-
(1-\delta)
\right|.
\]

### 7.6 Wireless Resource Cost

\[
c(a_t)
=
\lambda_p P_t
+
\lambda_b B_t
+
\lambda_l L_t
+
\lambda_e E_t
+
\lambda_c C_t.
\]

### 7.7 Unified Open Semantic Risk

\[
\boxed{
\mathcal{R}_{\mathrm{open}}(\pi)
=
\mathbb{E}
\left[
\ell_\tau(\hat{Y},Y)
+
\beta_1\ell_{\mathrm{unk}}
+
\beta_2\ell_{\mathrm{task}}
+
\beta_3\ell_{\mathrm{adapt}}
+
\beta_4\ell_{\mathrm{cal}}
+
\beta_5 c(a_t)
\right].
}
\]

This is the central metric and training/control objective of OpenSemCom.

---

## 8. Semantic Outage

### 8.1 Semantic Outage

A semantic outage occurs when the semantic output is accepted but violates a task-level semantic error threshold:

\[
\mathcal{O}_{\mathrm{sem}}
=
\left\{
d_t=\mathrm{accept},
\quad
\ell_\tau(\hat{Y}_t,Y_t)>\epsilon_\tau
\right\}.
\]

The semantic outage probability is:

\[
P_{\mathrm{so}}
=
\mathbb{P}
\left(
\mathcal{O}_{\mathrm{sem}}
\right).
\]

### 8.2 Open Semantic Outage

Open semantic outage includes task error, unknown-class acceptance, and task mismatch:

\[
P_{\mathrm{oso}}
=
\mathbb{P}
\left[
\ell_\tau(\hat{Y}_t,Y_t)>\epsilon_\tau
\ \lor\
\ell_{\mathrm{unk}}=1
\ \lor\
\ell_{\mathrm{task}}=1
\mid
d_t=\mathrm{accept}
\right].
\]

OpenSemCom enforces:

\[
P_{\mathrm{oso}} \le \delta.
\]

This gives semantic communication a wireless-style reliability metric analogous to outage probability, but defined over task-level meaning rather than symbol recovery.

---

## 9. OpenSemCom Architecture

OpenSemCom contains seven tightly coupled modules:

\[
\boxed{
\text{semantic parser}
+
\text{layered semantic encoder}
+
\text{wireless semantic channel}
+
\text{selective receiver}
+
\text{open-risk detector}
+
\text{safe adapter}
+
\text{risk-aware scheduler}.
}
\]

All modules optimize or estimate the same open semantic risk objective.

---

### 9.1 World-Aware Semantic Parser

The semantic parser maps raw source data to a task-relevant semantic representation:

\[
Z_t
=
f_\theta(X_t,\tau_t,c_t),
\]

where \(c_t\) includes context such as:

- channel state information;
- mobility state;
- location;
- user intent;
- task priority;
- device energy;
- receiver compute availability.

The parser decomposes the semantic representation into layers:

\[
Z_t
=
\left[
Z_t^{\mathrm{core}},
Z_t^{\mathrm{ref}},
Z_t^{\mathrm{evi}},
Z_t^{\mathrm{fb}}
\right].
\]

Where:

- \(Z_t^{\mathrm{core}}\): minimal task-relevant semantics;
- \(Z_t^{\mathrm{ref}}\): refinement semantics;
- \(Z_t^{\mathrm{evi}}\): evidence features for reliability checking;
- the paper-facing method does not use a fallback payload; insufficient reliability triggers semantic HARQ/refinement or reject/open.

A value-of-information score can be assigned to each semantic token:

\[
\mathrm{VoI}(z_i)
=
\mathbb{E}\!\left[
\mathcal{R}_{\mathrm{open}}
\mid z_i \text{ not transmitted}
\right]
-
\mathbb{E}\!\left[
\mathcal{R}_{\mathrm{open}}
\mid z_i \text{ transmitted}
\right].
\]

Tokens with high semantic value and high uncertainty receive higher transmission priority.

---

### 9.2 Layered Semantic Encoder

Instead of transmitting a single latent vector, OpenSemCom uses layered semantic transmission:

\[
\mathbf{x}_{t}^{(0)}
=
f_{\theta}^{\mathrm{core}}(Z_t^{\mathrm{core}}),
\]

\[
\mathbf{x}_{t}^{(1)}
=
f_{\theta}^{\mathrm{ref}}(Z_t^{\mathrm{ref}}),
\]

\[
\mathbf{x}_{t}^{(2)}
=
f_{\theta}^{\mathrm{evi}}(Z_t^{\mathrm{evi}}).
\]

The receiver first decodes the core layer. If the decoded semantic output is reliable, the system accepts. If uncertainty is high, it requests refinement or evidence.

This creates a semantic HARQ mechanism:

\[
\text{transmit core}
\rightarrow
\text{check risk}
\rightarrow
\text{transmit refinement/evidence if needed}.
\]

---

### 9.3 Selective Semantic Receiver

The receiver produces:

\[
(\hat{Y}_t,\Gamma_t,A_t,d_t),
\]

where:

- \(\hat{Y}_t\): point semantic prediction;
- \(\Gamma_t\): prediction set or semantic uncertainty set;
- \(A_t\): open nonconformity/risk score;
- \(d_t\): control decision.

The receiver decision is:

\[
d_t =
\begin{cases}
\mathrm{accept}, & A_t \le q_1,\\
\mathrm{refine}, & q_1 < A_t \le q_2,\\
\mathrm{reject/open}, & A_t > q_2.
\end{cases}
\]

Additional actions include semantic HARQ, safe adaptation, codec retrieval, and reject/open control.

---

### 9.4 Channel-Task-Aware Open-Risk Detector

The open-risk detector combines semantic, channel, task, and adaptation evidence:

\[
A_t
=
\alpha_1 A_{\mathrm{pred}}
+
\alpha_2 A_{\mathrm{proto}}
+
\alpha_3 A_{\mathrm{energy}}
+
\alpha_4 A_{\mathrm{recon}}
+
\alpha_5 A_{\mathrm{chan}}
+
\alpha_6 A_{\mathrm{task}}
+
\alpha_7 A_{\mathrm{adapt}}.
\]

Prediction uncertainty:

\[
A_{\mathrm{pred}}
=
1-\max_y p_\psi(y|\mathbf{r}_t,\tau_t).
\]

Prototype distance:

\[
A_{\mathrm{proto}}
=
\min_{c\in\mathcal{Y}_0}
\|z_t-\mu_c\|_2.
\]

Energy score:

\[
A_{\mathrm{energy}}
=
-\log
\sum_{y\in\mathcal{Y}_0}
\exp(g_y(z_t)).
\]

Semantic reconstruction inconsistency:

\[
A_{\mathrm{recon}}
=
d_{\mathrm{sem}}(X_t,\hat{X}_t).
\]

Channel shift score:

\[
A_{\mathrm{chan}}
=
D\!\left(
P_t(\mathbf{H})
\|P_0(\mathbf{H})
\right).
\]

Task mismatch score:

\[
A_{\mathrm{task}}
=
d(\tau_t,\mathcal{T}_0).
\]

Adaptation risk score:

\[
A_{\mathrm{adapt}}
=
\widehat{R}_{B_t}(\psi_t')-\widehat{R}_{B_t}(\psi_t).
\]

This detector is not a generic OOD detector. It is designed for semantic communication, where semantic uncertainty and wireless uncertainty interact.

---

### 9.5 Risk-Certified Safe Adaptation

The receiver has a frozen base model:

\[
\psi_0,
\]

and a lightweight adaptation module:

\[
\Delta_{\omega,t}.
\]

The deployed receiver is:

\[
\psi_t
=
\psi_0+\Delta_{\omega,t}.
\]

A candidate update is trained by:

\[
\Delta_{\omega,t}^{\prime}
=
\arg\min_{\Delta}
\left[
\mathcal{L}_{\mathrm{pseudo}}
+
\lambda_1\mathcal{L}_{\mathrm{cons}}
+
\lambda_2\|\Delta\|_2^2
+
\lambda_3\mathcal{L}_{\mathrm{cal}}
\right].
\]

The candidate receiver is:

\[
\psi_t'=\psi_0+\Delta_{\omega,t}'.
\]

The update is accepted only if:

\[
\widehat{R}_{B_t}(\psi_t')
+
\epsilon_t
\le
\widehat{R}_{B_t}(\psi_t)
-
\epsilon_t
-
\kappa,
\]

where:

\[
\epsilon_t
=
\sqrt{
\frac{\log(4T/\alpha)}
{2|B_t|}
}.
\]

If the gate fails:

\[
\psi_{t+1}=\psi_t.
\]

If the gate passes:

\[
\psi_{t+1}=\psi_t'.
\]

This prevents blind adaptation and reduces the risk of pseudo-label collapse.

---

### 9.6 Reliability-Card Semantic Codec Library

Each semantic codec module \(m\) has a reliability card:

\[
\mathcal{C}_m
=
\{
\mathcal{D}_m,
\mathcal{T}_m,
\mathcal{H}_m,
R_m(h,r,u),
P_{\mathrm{oso},m},
C_m,
L_m,
E_m,
\Gamma_m,
q_m
\}.
\]

Where:

- \(\mathcal{D}_m\): supported source domains;
- \(\mathcal{T}_m\): supported tasks;
- \(\mathcal{H}_m\): supported channel regimes;
- \(R_m(h,r,u)\): estimated open semantic risk;
- \(P_{\mathrm{oso},m}\): open semantic outage probability;
- \(C_m\): computation cost;
- \(L_m\): latency;
- \(E_m\): energy consumption;
- \(\Gamma_m\): conformal coverage certificate;
- \(q_m\): calibrated nonconformity threshold.

Codec routing solves:

\[
m_t^\star
=
\arg\min_{m\in\mathcal{M}}
\widehat{\mathcal{R}}_{\mathrm{open},m}
(h_t,r_t,u_t,\tau_t)
\]

subject to:

\[
P_{\mathrm{oso},m}\le\delta,
\]

\[
L_m\le L_{\max},
\]

\[
E_m\le E_{\max},
\]

\[
C_m\le C_{\max}.
\]

This turns codec selection into a reliability-aware decision rather than a similarity-only retrieval problem.

---

### 9.7 Risk-Aware Wireless Semantic Scheduler

The scheduler action is:

\[
a_t
=
\left(
p_t,
b_t,
r_t,
m_t,
\ell_t,
d_t
\right),
\]

where:

- \(p_t\): transmit power;
- \(b_t\): bandwidth;
- \(r_t\): semantic rate;
- \(m_t\): selected codec;
- \(\ell_t\): selected semantic layer;
- \(d_t\): receiver action.

For user \(k\), the scheduler solves:

\[
\min_{a_{k,t}}
\mathbb{E}
\left[
\mathcal{R}_{\mathrm{open},k}(a_{k,t})
\right]
\]

subject to:

\[
\sum_k p_{k,t}\le P_{\max},
\]

\[
\sum_k b_{k,t}\le B_{\max},
\]

\[
L_{k,t}\le L_k^{\max},
\]

\[
P_{\mathrm{oso},k}\le \delta_k.
\]

This makes wireless resource allocation central to OpenSemCom.

---

## 10. Optimization Formulation

The full optimization problem is:

\[
\min_{\theta,\psi,\phi,\eta}
\mathbb{E}
\left[
\ell_\tau
+
\beta_1\ell_{\mathrm{unk}}
+
\beta_2\ell_{\mathrm{task}}
+
\beta_3\ell_{\mathrm{adapt}}
+
\beta_4\ell_{\mathrm{cal}}
+
\beta_5c(a)
\right]
\]

subject to:

\[
P_{\mathrm{oso},k}\le \delta_k,
\quad
\forall k,
\]

\[
\sum_k p_k\le P_{\max},
\]

\[
\sum_k b_k\le B_{\max},
\]

\[
L_k\le L_k^{\max},
\]

\[
E_k\le E_k^{\max},
\]

\[
C_k\le C_k^{\max}.
\]

The Lagrangian is:

\[
\mathcal{L}
=
\mathcal{R}_{\mathrm{open}}
+
\sum_k \lambda_k
(P_{\mathrm{oso},k}-\delta_k)
+
\mu_p
\left(
\sum_k p_k-P_{\max}
\right)
+
\mu_b
\left(
\sum_k b_k-B_{\max}
\right)
+
\sum_k \mu_{l,k}
\left(
L_k-L_k^{\max}
\right)
+
\sum_k \mu_{e,k}
\left(
E_k-E_k^{\max}
\right).
\]

Primal-dual scheduling updates:

\[
\eta \leftarrow \eta-\gamma_\eta\nabla_\eta\mathcal{L},
\]

\[
\lambda_k
\leftarrow
\left[
\lambda_k+\gamma_\lambda(P_{\mathrm{oso},k}-\delta_k)
\right]^+,
\]

\[
\mu_p
\leftarrow
\left[
\mu_p+\gamma_p\left(\sum_k p_k-P_{\max}\right)
\right]^+.
\]

---

## 11. Algorithms

### Algorithm 1: OpenSemCom Inference Loop

**Input:** source \(X_t\), task \(\tau_t\), channel estimate \(\hat{H}_t\), resource budget \(\mathcal{B}_t\), codec library \(\mathcal{M}\), receiver \(\psi_t\).

1. Estimate channel state, semantic uncertainty, task context, and device resource state.
2. Select codec module \(m_t\) using reliability cards.
3. Parse source into layered semantic representation:
   \[
   Z_t=[Z_t^{\mathrm{core}},Z_t^{\mathrm{ref}},Z_t^{\mathrm{evi}},Z_t^{\mathrm{fb}}].
   \]
4. Transmit the core semantic layer.
5. Decode \(\hat{Y}_t\) and prediction set \(\Gamma_t\).
6. Compute open-risk score \(A_t\).
7. If \(A_t\le q_1\), accept the semantic output.
8. If \(q_1<A_t\le q_2\), request semantic refinement or semantic HARQ.
9. If \(A_t>q_2\), reject/open.
10. Update risk statistics and reliability-card estimates.

---

### Algorithm 2: Risk-Certified Safe Adaptation

**Input:** current receiver \(\psi_t\), candidate receiver \(\psi_t'\), buffer \(B_t\), confidence \(\alpha\), horizon \(T\), safety margin \(\kappa\).

1. Compute empirical risk:
   \[
   \widehat{R}_{B_t}(\psi_t),
   \quad
   \widehat{R}_{B_t}(\psi_t').
   \]
2. Compute confidence radius:
   \[
   \epsilon_t
   =
   \sqrt{
   \frac{\log(4T/\alpha)}
   {2|B_t|}
   }.
   \]
3. Accept update if:
   \[
   \widehat{R}_{B_t}(\psi_t')
   +
   \epsilon_t
   \le
   \widehat{R}_{B_t}(\psi_t)
   -
   \epsilon_t
   -
   \kappa.
   \]
4. Otherwise reject the update.
5. Keep the selective safety gate active at all times.

---

### Algorithm 3: Semantic HARQ / Refinement

**Input:** core layer, refinement layers, risk threshold \(q\), resource budget.

1. Transmit \(Z_t^{\mathrm{core}}\).
2. Receiver decodes and computes \(A_t^{(0)}\).
3. If:
   \[
   A_t^{(0)}\le q,
   \]
   accept.
4. Otherwise request \(Z_t^{\mathrm{ref}}\) or \(Z_t^{\mathrm{evi}}\).
5. Receiver updates prediction and computes:
   \[
   A_t^{(1)}.
   \]
6. Continue until:
   \[
   P_{\mathrm{oso}}\le \delta
   \]
   or resources are exhausted.
7. If reliability remains insufficient, reject/open.

---

### Algorithm 4: Risk-Aware Semantic Scheduling

**Input:** users \(k=1,\dots,K\), channel states \(\mathbf{H}_{k,t}\), semantic uncertainties \(u_{k,t}\), task priorities \(\rho_{k,t}\), budgets \(P_{\max},B_{\max}\).

1. Estimate risk reduction for each user and semantic layer:
   \[
   \Delta R_{k,i}.
   \]
2. Estimate communication and compute cost:
   \[
   c_{k,i}.
   \]
3. Rank semantic layers by:
   \[
   \frac{\Delta R_{k,i}}{c_{k,i}}.
   \]
4. Allocate resources using primal-dual updates under outage, power, bandwidth, latency, and compute constraints.
5. Trigger semantic HARQ or reject/open if calibrated risk remains above threshold.

---

## 12. Theoretical Analysis

The theoretical guarantees should be stated under explicit assumptions. OpenSemCom should not claim universal safety. It should claim finite-sample and conditional guarantees under well-defined statistical and wireless assumptions.

---

### 12.1 Assumptions

**A1. Bounded semantic loss**

\[
0\le \ell_\tau(\hat{Y},Y)\le 1.
\]

**A2. Calibration exchangeability or bounded shift**

Either calibration and deployment samples are exchangeable, or their density ratio is bounded:

\[
w(x,h,\tau)
=
\frac{dP_{\mathrm{test}}}{dP_{\mathrm{cal}}}
\le W_{\max}.
\]

**A3. Selective receiver**

The paper-facing receiver does not switch to a fallback decoder. Unsafe samples are refined through semantic HARQ or rejected/open.

**A4. Finite adaptation horizon**

There are at most \(T\) adaptation decisions.

**A5. Reliable validation or bounded pseudo-label noise**

The adaptation buffer \(B_t\) is labeled, delayed-labeled, or pseudo-labeled with bounded noise rate:

\[
\rho_t
=
\mathbb{P}(\tilde{Y}\ne Y)
\le \rho_{\max}.
\]

**A6. Semantic utility regularity**

Semantic risk reduction from additional layers is monotone and either concave or submodular.

---

### Theorem 1: Semantic Feasibility Lower Bound

Let \(Y\) be a finite semantic task variable and \(\tilde{Z}\) be the received semantic representation. If a decoder satisfies:

\[
P(\hat{Y}\ne Y)\le \epsilon,
\]

then:

\[
I(\tilde{Z};Y)
\ge
H(Y)
-
h(\epsilon)
-
\epsilon\log(|\mathcal{Y}|-1).
\]

Because the channel satisfies:

\[
I(Z;\tilde{Z})\le nC(H,P),
\]

a necessary condition for reliable semantic inference is:

\[
nC(H,P)
\ge
H(Y)
-
h(\epsilon)
-
\epsilon\log(|\mathcal{Y}|-1).
\]

**Interpretation.** If the channel cannot carry enough task-relevant information, no semantic decoder can guarantee low semantic task error. Therefore, the scheduler must sometimes allocate more power, bandwidth, redundancy, semantic refinement, or reject/open.

**Proof Sketch.** Fano's inequality gives:

\[
H(Y|\tilde{Z})
\le
h(\epsilon)+\epsilon\log(|\mathcal{Y}|-1).
\]

Since:

\[
I(\tilde{Z};Y)=H(Y)-H(Y|\tilde{Z}),
\]

the lower bound follows. The channel capacity upper bound follows from the data processing inequality and the channel capacity constraint.

---

### Theorem 2: Conformal Selective Semantic Reliability

Let \(A_i\) be calibration nonconformity scores and define:

\[
q_{1-\delta}
=
\mathrm{Quantile}_{1-\delta}(A_1,\dots,A_n).
\]

Define the prediction set:

\[
\Gamma(X,H,\tau)
=
\{y:A(X,H,\tau,y)\le q_{1-\delta}\}.
\]

Under exchangeability:

\[
\mathbb{P}
\left(
Y\notin \Gamma(X,H,\tau)
\right)
\le
\delta.
\]

Under weighted conformal calibration with estimated density ratios \(\hat{w}\):

\[
\mathbb{P}_{\mathrm{test}}
\left(
Y\notin \Gamma_w(X,H,\tau)
\right)
\le
\delta
+
O\left(
\|\hat{w}-w\|_1
+
\sqrt{\frac{1}{n_{\mathrm{eff}}}}
\right).
\]

**Selective-use note.** Ordinary conformal prediction gives marginal coverage. Conditional coverage after selection is not automatic. OpenSemCom therefore treats selection as part of the calibrated procedure and reports both:

\[
\mathbb{P}(Y\notin \Gamma)
\]

and:

\[
\mathbb{P}(Y\notin \Gamma \mid d=\mathrm{accept}).
\]

If the system calibrates the event:

\[
E =
\{\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\land d=\mathrm{accept}\},
\]

then:

\[
\mathbb{P}(E)\le\delta.
\]

If additionally:

\[
\mathbb{P}(d=\mathrm{accept})\ge \eta,
\]

then:

\[
\mathbb{P}(
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\mid d=\mathrm{accept}
)
\le
\frac{\delta}{\eta}.
\]

This links conformal reliability to open semantic outage.

---

### Theorem 3: Safe Adaptation Non-Degradation by Induction

Let \(\psi_t\) be the current receiver and \(\psi_t'\) be a candidate adapted receiver. Define empirical risk on buffer \(B_t\):

\[
\widehat{R}_{B_t}(\psi)
=
\frac{1}{|B_t|}
\sum_{(x_i,y_i)\in B_t}
\ell(q_\psi(x_i),y_i).
\]

The adaptation gate is:

\[
\widehat{R}_{B_t}(\psi_t')
+
\epsilon_t
\le
\widehat{R}_{B_t}(\psi_t)
-
\epsilon_t
-
\kappa,
\]

where:

\[
\epsilon_t
=
\sqrt{
\frac{\log(4T/\alpha)}
{2|B_t|}
}.
\]

If \(B_t\) contains true or delayed labels sampled from the current deployment distribution, then with probability at least \(1-\alpha\), every accepted adaptation satisfies:

\[
R_t(\psi_{t+1})
\le
R_t(\psi_t)-\kappa.
\]

If pseudo-labels are used with noise rate \(\rho_t\), then:

\[
R_t(\psi_{t+1})
\le
R_t(\psi_t)-\kappa+2\rho_t.
\]

Thus adaptation is safe when:

\[
\kappa>2\rho_t.
\]

**Proof by Induction.**

Base case: At \(t=0\), the deployed model is the frozen receiver \(\psi_0\). No adaptation harm has occurred.

Induction hypothesis: Assume that up to time \(t\), every accepted update has satisfied the certified non-degradation condition.

Induction step: Hoeffding's inequality gives, with probability at least \(1-\alpha/T\):

\[
|R_t(\psi_t)-\widehat{R}_{B_t}(\psi_t)|\le \epsilon_t,
\]

and:

\[
|R_t(\psi_t')-\widehat{R}_{B_t}(\psi_t')|\le \epsilon_t.
\]

If the gate accepts:

\[
R_t(\psi_t')
\le
\widehat{R}_{B_t}(\psi_t')+\epsilon_t,
\]

\[
\le
\widehat{R}_{B_t}(\psi_t)-\epsilon_t-\kappa,
\]

\[
\le
R_t(\psi_t)-\kappa.
\]

Therefore:

\[
R_t(\psi_{t+1})\le R_t(\psi_t)-\kappa.
\]

If the gate rejects:

\[
\psi_{t+1}=\psi_t,
\]

so adaptation does not increase risk. Applying a union bound over \(T\) rounds gives probability at least \(1-\alpha\).

For pseudo-labels, the true and pseudo-label risks differ by at most \(\rho_t\) for each model, so the comparison incurs an additional \(2\rho_t\) term.

---

### Theorem 4: Risk-Aware Semantic Scheduling Optimality

Assume semantic layer \(i\) provides expected risk reduction:

\[
g_i(r_i,h_i,u_i),
\]

where:

- \(r_i\): allocated semantic rate;
- \(h_i\): channel quality;
- \(u_i\): semantic uncertainty.

Assume \(g_i\) is increasing and concave in \(r_i\). The scheduler solves:

\[
\max_{r_i\ge0}
\sum_i g_i(r_i,h_i,u_i)
-
\lambda\sum_i r_i
\]

subject to:

\[
\sum_i r_i\le R_{\max}.
\]

For active layers, the KKT condition is:

\[
\frac{\partial g_i(r_i,h_i,u_i)}{\partial r_i}
=
\lambda.
\]

**Interpretation.** The scheduler allocates semantic rate until marginal semantic risk reduction equals marginal resource cost. High-uncertainty and high-importance semantic layers receive more resources because their marginal risk reduction is larger.

---

### Theorem 5: Greedy Semantic-Layer Selection Guarantee

Let \(G(S)\) be the total risk reduction from selected semantic layers \(S\). If \(G\) is monotone submodular and each layer has cost \(c_i\), then greedy selection by:

\[
\frac{\Delta G_i}{c_i}
\]

under budget:

\[
\sum_{i\in S}c_i\le C_{\max}
\]

achieves:

\[
G(S_{\mathrm{greedy}})
\ge
\left(
1-\frac{1}{e}
\right)
G(S^\star).
\]

**Interpretation.** Semantic HARQ and refinement-layer scheduling can have a standard approximation guarantee when marginal risk reduction is submodular.

---

### Theorem 6: Open Semantic Outage Bound

Define accepted open semantic outage:

\[
P_{\mathrm{oso}}
=
\mathbb{P}
\left[
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\mid d=\mathrm{accept}
\right].
\]

If the calibrated acceptance procedure guarantees:

\[
\mathbb{P}
\left[
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\land
d=\mathrm{accept}
\right]
\le \delta,
\]

and the system enforces:

\[
\mathbb{P}(d=\mathrm{accept})\ge \eta,
\]

then:

\[
P_{\mathrm{oso}}
\le
\frac{\delta}{\eta}.
\]

**Interpretation.** Open semantic outage can be bounded by calibrating the accepted-error event and controlling the acceptance rate.

---

### Theorem 7: Latent Shift Risk Bound

Let \(P_0(Z,Y)\) be the training latent distribution and \(P_t(Z,Y)\) be the deployment latent distribution. For receiver \(q_\psi\), a domain adaptation-style bound can be written as:

\[
R_t(q_\psi)
\le
R_0(q_\psi)
+
\mathrm{IPM}(P_0(Z),P_t(Z))
+
\lambda^\star,
\]

where:

- \(\mathrm{IPM}\) is an integral probability metric;
- \(\lambda^\star\) is the error of the best shared hypothesis over both domains.

**Interpretation.** If the semantic encoder maps shifted domains into stable task-relevant latent distributions, deployment risk remains bounded. If latent mismatch grows, OpenSemCom should trigger refinement, adaptation, or rejection/open.

---

## 13. Training Methodology

### 13.1 Full Training Objective

\[
\mathcal{L}_{\mathrm{train}}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_1\mathcal{L}_{\mathrm{sem-rec}}
+
\lambda_2\mathcal{L}_{\mathrm{IB}}
+
\lambda_3\mathcal{L}_{\mathrm{contrast}}
+
\lambda_4\mathcal{L}_{\mathrm{open}}
+
\lambda_5\mathcal{L}_{\mathrm{cal}}
+
\lambda_6\mathcal{L}_{\mathrm{DRO}}
+
\lambda_7\mathcal{L}_{\mathrm{resource}}.
\]

### 13.2 Task Loss

\[
\mathcal{L}_{\mathrm{task}}
=
\ell_\tau(\hat{Y},Y).
\]

This can be classification cross-entropy, detection loss, segmentation loss, retrieval loss, or control loss.

### 13.3 Semantic Reconstruction Loss

\[
\mathcal{L}_{\mathrm{sem-rec}}
=
d_{\mathrm{sem}}(X,\hat{X}),
\]

where \(d_{\mathrm{sem}}\) may combine perceptual distance, feature distance, and task-level semantic consistency.

### 13.4 Information Bottleneck Loss

\[
\mathcal{L}_{\mathrm{IB}}
=
I(X;Z|H,\tau)
-
\gamma I(Z;Y_\tau|H,\tau).
\]

This encourages \(Z\) to be compact while preserving task-relevant semantics.

### 13.5 Contrastive Semantic Alignment

\[
\mathcal{L}_{\mathrm{contrast}}
=
-\log
\frac{
\exp(\mathrm{sim}(z_i,z_i^+)/\tau_c)
}{
\sum_j \exp(\mathrm{sim}(z_i,z_j)/\tau_c)
}.
\]

This improves semantic clustering and open-set separation.

### 13.6 Open-Risk Detector Loss

\[
\mathcal{L}_{\mathrm{open}}
=
\mathcal{L}_{\mathrm{BCE}}(o,\hat{o})
+
\lambda_m
\max(0,m-d_{\mathrm{proto}})
+
\lambda_e
\mathcal{L}_{\mathrm{energy}}.
\]

### 13.7 Calibration Loss

\[
\mathcal{L}_{\mathrm{cal}}
=
\mathrm{ECE}
+
\lambda_{\Gamma}
|\Gamma|.
\]

This encourages calibrated confidence while avoiding excessively large prediction sets.

### 13.8 Robustness Loss

A distributionally robust objective can be written as:

\[
\mathcal{L}_{\mathrm{DRO}}
=
\sup_{Q\in\mathcal{U}(P_0)}
\mathbb{E}_{Q}
[
\ell_\tau+\beta \ell_{\mathrm{open}}
].
\]

### 13.9 Resource Loss

\[
\mathcal{L}_{\mathrm{resource}}
=
\lambda_pP_t+\lambda_bB_t+\lambda_lL_t+\lambda_eE_t+\lambda_cC_t.
\]

---

## 14. OpenSemCom-Bench

OpenSemCom-Bench should evaluate semantic communication under controlled combinations of source, channel, semantic, task, supervision, and resource shifts.

### 14.1 Evaluation Regimes

| Regime | Description |
|---|---|
| **R1: Closed-ID** | Known source, known classes, known task, known channel distribution. |
| **R2: Channel-Open** | Known semantic distribution but shifted channel, fading, interference, CSI error, blockage, or SNR. |
| **R3: Source-Open** | New visual/sensor domain, weather, lighting, mobility, camera, environment, or device condition. |
| **R4: Class-Open** | Unseen semantic classes appear at deployment. |
| **R5: Task-Open** | New downstream task or changed task utility. |
| **R6: Supervision-Limited** | Few labeled samples, delayed labels, or unlabeled edge data only. |
| **R7: Resource-Open** | Power, bandwidth, latency, energy, or compute budget differs from training. |
| **R8: Full-Open** | Source shift + channel shift + class novelty + task drift + supervision limitation + resource constraint. |

The main stress test is **R8: Full-Open**.

---

### 14.2 Data Modalities and Tasks

#### Image Semantic Communication

Datasets:

- CIFAR-10 / CIFAR-100;
- ImageNet subsets;
- COCO.

Tasks:

- classification;
- object detection;
- semantic reconstruction;
- retrieval.

#### Smart City and Autonomous Driving

Datasets:

- Cityscapes;
- BDD100K;
- nuScenes.

Tasks:

- semantic segmentation;
- object detection;
- scene understanding;
- hazard prediction.

#### Multimodal Wireless Sensing

Datasets:

- DeepSense 6G, including field-collected multimodal sensing and communication data. [DeepSense6G][12]

Modalities:

- RGB camera;
- LiDAR;
- radar;
- GPS;
- mmWave beam measurements.

Tasks:

- beam prediction;
- blockage prediction;
- scene recognition;
- multimodal semantic inference.

#### Text and Language Semantics

Datasets:

- text classification corpora;
- sentence similarity corpora;
- task-oriented command datasets.

Tasks:

- intent detection;
- semantic similarity;
- instruction recovery;
- task-oriented semantic transmission.

---

### 14.3 Wireless Channel Models

OpenSemCom-Bench should include:

\[
\text{AWGN},
\]

\[
\text{Rayleigh fading},
\]

\[
\text{Rician fading},
\]

\[
\text{Nakagami fading},
\]

\[
\text{MIMO},
\]

\[
\text{OFDM},
\]

\[
\text{mmWave blockage},
\]

\[
\text{Doppler shift},
\]

\[
\text{co-channel interference},
\]

\[
\text{CSI estimation error},
\]

\[
\text{burst noise},
\]

\[
\text{jamming/interference}.
\]

The benchmark should vary not only SNR, but also the distribution of wireless impairments:

\[
P_{\mathrm{train}}(H)\ne P_{\mathrm{test}}(H).
\]

---

### 14.4 Open-Class and Source-Shift Protocol

For class-open evaluation:

\[
\mathcal{Y}_{\mathrm{train}}\cap\mathcal{Y}_{\mathrm{unknown}}=\emptyset.
\]

The system is trained on known classes and tested on a mixture of known and unknown classes.

For source-open evaluation:

\[
P_{\mathrm{train}}(X)\ne P_{\mathrm{test}}(X),
\]

using changes in weather, lighting, camera type, environment, sensor modality, or scene distribution.

For task-open evaluation:

\[
\tau_{\mathrm{test}}\notin\mathcal{T}_{\mathrm{train}},
\]

or the utility function changes:

\[
u_{\mathrm{test}}(Y,\hat{Y})\ne u_{\mathrm{train}}(Y,\hat{Y}).
\]

---

## 15. Baselines

OpenSemCom should be compared against:

1. Classical digital communication: source coding + channel coding.
2. DeepJSCC-style wireless image transmission. [DeepJSCC][1]
3. DeepSC-style text semantic communication. [DeepSC][2]
4. WITT-style wireless image semantic transmission. [WITT][3]
5. Task-agnostic semantic communication, such as TASC. [TASC][4]
6. Foundation-model token semantic communication, such as SemCLIP. [SemCLIP][5]
7. Distributionally robust semantic communication, such as WaSeCom. [WaSeCom][6]
8. Conformal semantic communication under channel shift. [ConformalSemCom][7]
9. Semantic receiver with generic OOD detector.
10. Semantic receiver with naive test-time adaptation.
11. Resource-aware semantic scheduler without certification.
12. OpenSemCom without safe adaptation.
13. OpenSemCom without semantic HARQ.
14. Full OpenSemCom.

---

## 16. Metrics

### 16.1 Communication Metrics

\[
\mathrm{BER},
\quad
\mathrm{BLER},
\quad
\mathrm{SNR},
\quad
\mathrm{spectral\ efficiency},
\quad
\mathrm{outage\ probability}.
\]

### 16.2 Reconstruction Metrics

\[
\mathrm{PSNR},
\quad
\mathrm{SSIM},
\quad
\mathrm{LPIPS}.
\]

### 16.3 Semantic Task Metrics

\[
\mathrm{Accuracy},
\quad
\mathrm{F1},
\quad
\mathrm{mAP},
\quad
\mathrm{mIoU},
\quad
\mathrm{Recall@K}.
\]

### 16.4 Open-World Metrics

\[
\mathrm{AUROC}_{\mathrm{OOD}},
\quad
\mathrm{FPR95},
\quad
\mathrm{AUPR},
\quad
\mathrm{OpenSetF1}.
\]

### 16.5 Calibration Metrics

\[
\mathrm{ECE},
\quad
\mathrm{Brier},
\quad
\mathrm{Coverage},
\quad
\mathrm{PredictionSetSize}.
\]

### 16.6 Adaptation Safety Metrics

\[
\mathrm{AdaptationHarmRate}
=
\frac{1}{T}
\sum_t
\mathbf{1}
\left[
R_t(\psi_t)>R_t(\psi_{t-1})
\right].
\]

\[
\mathrm{CertifiedAcceptRate}
=
\frac{
\#\text{accepted adaptations}
}{
\#\text{candidate adaptations}
}.
\]

\[
\mathrm{SafeGain}
=
R(\psi_0)-R(\psi_T).
\]

### 16.7 Wireless Semantic Reliability Metrics

\[
P_{\mathrm{so}}
=
\mathbb{P}
\left[
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\right].
\]

\[
P_{\mathrm{oso}}
=
\mathbb{P}
\left[
\ell_\tau(\hat{Y},Y)>\epsilon_\tau
\mid d=\mathrm{accept}
\right].
\]

\[
\mathrm{SemanticGoodput}
=
\frac{
\#\text{correct accepted semantic decisions}
}{
\#\text{channel uses}
}.
\]

\[
\mathrm{RiskPerJoule}
=
\frac{
\mathcal{R}_{\mathrm{open}}
}{
E
}.
\]

\[
\mathrm{RiskLatencyProduct}
=
\mathcal{R}_{\mathrm{open}}L.
\]

The main metric is:

\[
\boxed{
\mathcal{R}_{\mathrm{open}}.
}
\]

---

## 17. Ablation Studies

| Ablation | Purpose |
|---|---|
| No open-risk detector | Measures the value of open-environment detection. |
| No conformal calibration | Measures the value of finite-sample reliability calibration. |
| No safe adaptation gate | Shows whether naive adaptation causes semantic harm. |
| Full fine-tuning instead of adapters | Tests edge cost and adaptation instability. |
| No semantic HARQ/refinement | Measures the value of progressive reliability control. |
| Fixed-rate scheduler | Tests benefit of risk-aware resource allocation. |
| No codec reliability cards | Tests benefit of certified module selection. |
| No channel-aware detector term | Tests wireless-specific contribution. |
| No task-shift detector term | Tests task-open reliability. |
| No abstention | Shows the danger of forced semantic prediction. |
| No pseudo-label noise control | Tests the safety theorem's noise sensitivity. |
| No resource constraints | Isolates semantic accuracy from wireless resource efficiency. |

The most important ablation is:

\[
\text{safe adaptation gate}
\quad
\text{versus}
\quad
\text{naive pseudo-label adaptation}.
\]

---

## 18. Expected Empirical Claims

The experiments should support the following claims.

### Claim 1: Lower Open Semantic Risk

OpenSemCom should reduce:

\[
\mathcal{R}_{\mathrm{open}}
\]

under full-open deployment compared with standard semantic communication and robust semantic communication baselines.

### Claim 2: Lower Open Semantic Outage

OpenSemCom should reduce:

\[
P_{\mathrm{oso}}
\]

under channel shift, source shift, class novelty, and task drift.

### Claim 3: Safer Adaptation

OpenSemCom should reduce:

\[
\mathrm{AdaptationHarmRate}
\]

relative to naive test-time adaptation.

### Claim 4: Better Reliability-Resource Tradeoff

Semantic HARQ and risk-aware scheduling should improve:

\[
\mathrm{SemanticGoodput},
\quad
\mathrm{RiskPerJoule},
\quad
\mathrm{RiskLatencyProduct}.
\]

### Claim 5: Better Open-Class Handling

The open-risk detector should improve:

\[
\mathrm{AUROC}_{\mathrm{OOD}},
\quad
\mathrm{FPR95},
\quad
\mathrm{OpenSetF1}.
\]

### Claim 6: More Stable Operation Under Multi-Axis Shift

The full system should degrade gracefully under simultaneous source, channel, class, task, supervision, and resource shifts.

---

## 19. Contribution Statements

OpenSemCom makes the following contributions.

### Contribution 1: Open-Environment Semantic Communication Formulation

OpenSemCom formulates semantic communication under simultaneous source shift, channel shift, semantic class novelty, task drift, limited supervision, and wireless edge-resource constraints.

### Contribution 2: Open Semantic Risk

OpenSemCom introduces open semantic risk, a unified metric combining task error, unknown-class acceptance, task mismatch, adaptation harm, calibration error, and wireless resource cost.

### Contribution 3: Open Semantic Outage

OpenSemCom defines semantic outage and open semantic outage as wireless reliability metrics for task-level meaning.

### Contribution 4: Selective Semantic Receiver

OpenSemCom designs a receiver that outputs a semantic prediction, uncertainty set, open-risk score, and reliability-control action.

### Contribution 5: Risk-Certified Safe Adaptation

OpenSemCom proposes a lightweight receiver adaptation method with a finite-sample acceptance gate and a high-probability non-degradation guarantee.

### Contribution 6: Semantic HARQ and Reliability-Aware Refinement

OpenSemCom introduces semantic HARQ/refinement, where additional semantic layers are transmitted only when calibrated risk is too high.

### Contribution 7: Reliability-Card Codec Library

OpenSemCom develops a modular semantic codec library with reliability cards specifying supported domains, tasks, channels, cost profiles, risk curves, and calibration certificates.

### Contribution 8: Risk-Aware Wireless Semantic Scheduling

OpenSemCom formulates resource control as constrained open-risk minimization over power, bandwidth, semantic rate, latency, energy, compute, codec selection, and semantic layer selection.

### Contribution 9: OpenSemCom-Bench

OpenSemCom provides a benchmark protocol for evaluating semantic communication under controlled combinations of source, channel, class, task, supervision, and resource shifts.

---

## 20. Paper Structure

### Section I: Introduction

Main points:

- Semantic communication is promising for 6G.
- Static and closed-world assumptions are fragile.
- Semantic failure under open environments can be more dangerous than bit errors.
- OpenSemCom provides risk-certified selective semantic communication.

### Section II: Related Work

Organize around:

1. learned JSCC and semantic transmission;
2. task-oriented and task-agnostic semantic communication;
3. robust and distributionally robust semantic communication;
4. conformal semantic communication;
5. OOD detection and open-set recognition;
6. test-time adaptation;
7. semantic resource allocation and wireless scheduling.

### Section III: System Model and Problem Formulation

Include:

- multi-user wireless channel;
- semantic source and task model;
- open-environment definition;
- open semantic risk;
- semantic outage;
- constrained optimization problem.

### Section IV: OpenSemCom Architecture

Include:

- world-aware semantic parser;
- layered semantic encoder;
- selective semantic receiver;
- open-risk detector;
- risk-certified safe adaptation;
- reliability-card codec library;
- risk-aware scheduler.

### Section V: Theoretical Analysis

Include:

1. semantic feasibility lower bound;
2. conformal selective semantic reliability;
3. safe adaptation induction theorem;
4. risk-aware scheduling KKT condition;
5. greedy semantic-layer guarantee;
6. open semantic outage bound;
7. latent shift risk bound.

### Section VI: Algorithms

Include:

- OpenSemCom inference loop;
- safe adaptation algorithm;
- semantic HARQ/refinement algorithm;
- primal-dual scheduling algorithm.

### Section VII: Experimental Setup

Include:

- datasets;
- tasks;
- channel models;
- open-environment regimes;
- baselines;
- metrics.

### Section VIII: Results

Include:

- closed-ID results;
- channel-open results;
- source-open results;
- class-open results;
- task-open results;
- full-open results;
- resource tradeoffs;
- adaptation safety;
- calibration and outage;
- ablation studies.

### Section IX: Conclusion

Emphasize:

- open-environment semantic reliability;
- safe adaptation;
- semantic outage-aware wireless communication;
- risk-aware resource allocation;
- future deployment in 6G edge intelligence.

---

## 21. Implementation Plan

### Phase 1: Core Prototype

Build:

- encoder/decoder over AWGN and Rayleigh channels;
- semantic receiver with task head;
- open-risk detector;
- conformal calibration;
- basic semantic outage evaluation.

Target tasks:

- image classification;
- semantic reconstruction;
- open-set detection.

### Phase 2: Safe Adaptation

Add:

- lightweight adapters;
- pseudo-label filtering;
- consistency regularization;
- safe adaptation gate;
- adaptation harm metrics.

### Phase 3: Semantic HARQ and Scheduling

Add:

- core/refinement/evidence semantic layers;
- semantic HARQ protocol;
- risk-aware resource scheduler;
- power/bandwidth/latency constraints.

### Phase 4: Multi-User Wireless System

Add:

- MIMO channel;
- interference;
- multi-user scheduling;
- semantic goodput evaluation;
- open semantic outage constraints.

### Phase 5: OpenSemCom-Bench

Build full benchmark with:

- source shifts;
- channel shifts;
- class novelty;
- task drift;
- supervision limitation;
- resource changes;
- full-open deployment.

---

## 22. Reviewer-Focused Technical Positioning

The paper should be positioned as:

> OpenSemCom is a wireless semantic reliability and resource-control framework. It formulates open-environment semantic communication as constrained open-risk minimization and provides selective reliability, safe adaptation, semantic outage analysis, semantic HARQ, and risk-aware scheduling.

The paper should avoid being perceived as merely:

\[
\text{semantic communication}
+
\text{OOD detection}
+
\text{adaptation}
+
\text{scheduling}.
\]

Instead, the core technical spine should be:

\[
\boxed{
\text{open semantic risk}
\rightarrow
\text{selective semantic reliability}
\rightarrow
\text{safe adaptation}
\rightarrow
\text{semantic outage}
\rightarrow
\text{wireless resource control}.
}
\]

---

## 23. Potential Limitations and Mitigation

| Risk | Mitigation |
|---|---|
| Too many modules | Emphasize the unified open-risk objective and make safe adaptation + semantic outage + scheduling the central spine. |
| Conditional coverage is difficult | Report marginal coverage, accepted-error probability, and conditional outage; calibrate accepted-error events explicitly. |
| Pseudo-label adaptation may be unsafe | Use bounded-noise assumptions, delayed labels when available, and conservative gates. |
| Wireless channel may seem secondary | Include MIMO, interference, CSI error, semantic HARQ, outage, and power/bandwidth scheduling. |
| Benchmark may be too broad | Start with image + multimodal sensing, then extend to text and multi-task settings. |
| Theoretical guarantees may look idealized | State assumptions clearly and validate assumption sensitivity experimentally. |
| Foundation models may be computationally heavy | Use lightweight adapters, distillation, token pruning, and reliability-card compute constraints. |

---

## 24. Final Abstract

**OpenSemCom: Risk-Certified Open-Environment Semantic Communication for Dynamic 6G Wireless Networks**

Semantic communication aims to improve communication efficiency by transmitting task-relevant meaning rather than raw bits. However, many semantic communication systems are designed under closed or partially controlled assumptions, where the source distribution, semantic label space, task objective, wireless channel condition, and edge-resource budget are known or only mildly varying. Such assumptions are fragile in dynamic 6G wireless networks, where devices must operate under changing channels, source-domain shifts, unseen semantic classes, new downstream tasks, limited labels, interference, latency constraints, and edge computation limits. Under these conditions, a semantic receiver may produce confident but incorrect predictions, generative semantic decoders may hallucinate plausible but wrong content, and test-time adaptation may degrade reliability.

This paper proposes OpenSemCom, a risk-certified open-environment semantic communication framework for dynamic 6G wireless networks. OpenSemCom formulates semantic communication as a multi-axis reliability problem involving channel shift, source shift, class novelty, task drift, limited supervision, and resource constraints. It introduces open semantic risk, a unified objective that combines task error, unknown-class acceptance, task mismatch, adaptation harm, calibration failure, and wireless resource cost. OpenSemCom integrates a layered semantic encoder, a selective semantic receiver, a channel-task-aware open-risk detector, conformal reliability calibration, lightweight safe adaptation, semantic HARQ/refinement, a reliability-card codec library, and risk-aware wireless scheduling. The paper-facing receiver does not use a fallback decoder: it accepts when risk is below \(q_1\), refines or requests semantic HARQ when \(q_1 <\) risk \(\le q_2\), and rejects/open otherwise.

Theoretically, OpenSemCom derives semantic feasibility lower bounds connecting task reliability to mutual information and channel capacity, provides finite-sample selective reliability guarantees, proves high-probability non-degradation for accepted adaptation updates using induction, derives optimality conditions for risk-aware semantic scheduling, and establishes open semantic outage bounds. Experimentally, OpenSemCom-Bench evaluates image, multimodal sensing, and wireless edge tasks under controlled source, channel, class, task, supervision, and resource shifts. OpenSemCom is expected to reduce open semantic risk, lower semantic outage, prevent unsafe adaptation, and improve semantic goodput compared with conventional JSCC, semantic communication, robust semantic communication, conformal semantic communication, adaptation-based, and resource-aware baselines.

---

## 25. Final Title

**OpenSemCom: Risk-Certified Open-Environment Semantic Communication for Dynamic 6G Wireless Networks**

---

## References

[1]: https://arxiv.org/abs/1809.01733 "Deep Joint Source-Channel Coding for Wireless Image Transmission"

[2]: https://arxiv.org/abs/2006.10685 "Deep Learning Enabled Semantic Communication Systems"

[3]: https://arxiv.org/abs/2211.00937 "WITT: A Wireless Image Transmission Transformer for Semantic Communications"

[4]: https://arxiv.org/abs/2504.21723 "Task-Agnostic Semantic Communications Relying on Information Bottleneck and Federated Meta-Learning"

[5]: https://arxiv.org/html/2502.18200v1 "Task-Agnostic Semantic Communication with Multimodal Foundation Models"

[6]: https://arxiv.org/abs/2506.03167 "Distributionally Robust Wireless Semantic Communication with Large AI Models"

[7]: https://openreview.net/forum?id=M4xtV1weHZ "Conformal Semantic Communication: Distribution-Free Task-Level Coverage Guarantees for Goal-Oriented Transmission Under Channel Shift"

[8]: https://link.springer.com/article/10.1007/s11263-024-02222-4 "Dissecting Out-of-Distribution Detection and Open-Set Recognition"

[9]: https://link.springer.com/article/10.1007/s11263-024-02213-5 "In Search of Lost Online Test-Time Adaptation: A Survey"

[10]: https://proceedings.mlr.press/v202/zhao23d/zhao23d.pdf "On Pitfalls of Test-Time Adaptation"

[11]: https://www.itu.int/hub/2026/03/imt-2030-technical-requirements-for-the-6g-future/ "IMT-2030: Technical requirements for the 6G future"

[12]: https://arxiv.org/pdf/2211.09769 "DeepSense 6G"

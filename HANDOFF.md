# AutoMaX (`amx`) — Claude Code Handoff

| | |
|---|---|
| Project | **AutoMaX**, package and CLI shortform **`amx`** |
| Doc version | 0.1.1 (handoff + domain-neutral wording, D17), 2026-10-05. Open questions: `docs/OQ.md`. Build plan: `docs/ROLLER.md`. |
| Audience | Claude Code (builder) and the owner (decision-maker) |
| Status | Phase A design locked for build. Phase B is an interface contract only. |
| One-liner | Given a labelled dataset, a task spec and acceptable risk bands, Claude Code designs an **LLM-free-at-inference** pipeline (any mix of classical ML / NLP / DL / CV / stats), certifies thresholds and coverage per band, and hands the unresolved residual, with reasons, to a later LLM phase. |

## Read-first brief (60 seconds)

1. AutoMaX takes labelled data + a TaskSpec + risk bands and returns a **frozen, deterministic, LLM-free resolver**, a **certified risk-coverage report**, and a **handoff manifest** for the unresolved residual.
2. Everything is expressed as **commit unit + loss ℓ ∈ [0,1] + bands**. One harness serves classification, regression, forecasting, extraction and anomaly tasks (release R1). Multi-label, ranking and detection follow (R2).
3. Inside is a **typed graph of operators** (rules, stats, ML, DL, verifiers, calibrators, routers, cascades, loops), each emitting a value and uncertainty signals. A learned scorer turns signals into a predicted loss; a single threshold τ gives a nested commit rule.
4. **Certification** is fixed-sequence Learn-then-Test along τ, using calibration data that only a **warden** process can read. Guarantee types are labelled honestly; forecasting is weaker.
5. Claude Code **designs and searches** (proposer, auditor, analyst, reporter). It never grades and never touches calibration or sealed data. Acceptance is a deterministic evaluator plus a noise margin.
6. **Domain enters through four slots** (data, ℓ, constraints, priors). A lint enforces domain-blindness.
7. Build **A0 to A4** behind gates. A0 proves the statistics on synthetic data before any agent exists.
8. Tests **T1 to T5**: certificate validity, earns complexity against best-single + same certifier, reason codes validated on public ground truth, portability and cost, leave-one-domain-out.
9. Do not write an evolutionary controller. Plug the evaluator into a simple ratchet or an existing LLM-evolution framework, and A/B it against TPOT-style and bandit baselines.
10. Open decisions O1 to O10 are in section 17. Defaults exist, so A0 can start immediately.

---

## Table of contents

0. How to use this document
1. Mission, scope, non-goals
2. Core abstraction
3. Task families and release sequencing
4. Architecture and repo layout
5. Run directory, artifacts, CLI
6. Contracts (TaskSpec, Operator, Graph, Evaluator, Report, Manifest, Ledger)
7. Statistics (normative)
8. Splits, leakage, and sealed-fold protection
9. Search (backends, fitness, supervision)
10. Reason codes, verifiers, analyst, handoff to Phase B
11. Claude Code plugin design
12. Acceptance tests T1 to T5
13. Validation datasets
14. Milestones A0 to A4
15. Dev environment, tooling, test strategy
16. Risks and traps
17. Open decisions and decision log
18. Kickoff: checklist, prompt, CLAUDE.md starter
19. References
- Appendix A: pseudocode and formulas · B: example TaskSpecs · C: glossary

---

## 0. How to use this document

1. Read sections 1 to 8 before writing code. Sections 6 and 7 are **normative contracts**: implement as written. Deviations need an entry in the decision log (section 17).
2. Build strictly in milestone order (section 14). Every milestone has a **gate**. Do not start milestone N+1 until N's gate passes.
3. **Honesty rules.** Every certificate must state its guarantee type (section 7.1) and its assumptions. Never emit the word "certified" for a number whose assumptions were not audited.
4. Anything marked **VERIFY** must be checked against the live paper or docs before implementing. Claude Code features change quickly; statistical formulas must be unit-tested against Monte Carlo.
5. Owner decisions are collected in section 17. Ask for them at kickoff instead of guessing.
6. This document was written from a design conversation. Where it says "locked", treat as decided. Where it says "initial", the number is a starting guess to be tuned and recorded in the decision log.

---

## 1. Mission, scope, non-goals

### 1.1 Mission

Given:
- a **labelled dataset** (structured or unstructured, any domain),
- a **TaskSpec** (what a "unit" is, how a committed answer is scored against gold, which risk bands are acceptable),

AutoMaX automatically:
1. **Designs** a pipeline with Claude Code workflows as the designer. The pipeline may combine rules, statistical models, classical ML, NLP/CV featurizers, small deep models, probabilistic calibration, verifiers, routers, cascades and loops. It is searched as a typed graph.
2. **Certifies**, for each requested risk band, a confidence threshold and the coverage achievable under it, with finite-sample statistical guarantees (and honest labels when only weaker guarantees are possible).
3. **Characterizes the residual** (units it should not resolve), assigns reason codes, and writes a handoff manifest recommending which residual segments an LLM may help with, which need a human, and which are irreducible.

The product is a **certified risk-coverage frontier plus a frozen, deterministic resolver**, not "a model".

### 1.2 Two phases

| Phase | What | Status |
|---|---|---|
| **A** (this build) | Design and certify the non-LLM pipeline ("Set A"); characterize and hand off the residual ("Set B"). | Build now |
| **B** (later) | Consume the manifest; design the LLM-side handling of Set B: retrieval of exemplars, rubric injection, candidate-set restriction, self-consistency, tool use, etc. | Out of scope. Only the manifest contract (section 10.5) is in scope. |

### 1.3 "LLM-free at inference" means

- The frozen artifact never calls a generative model.
- Claude Code is **design-time only**: proposer, analyst, auditor, reporter.
- Frozen pretrained encoders (sentence-transformers, DINOv2/CLIP-style) **are allowed** (decision D5). They are not generative. Their identity and version are recorded, and a contamination caveat is reported (section 16).

### 1.4 Non-goals for v1

- No generative tasks, no unlabeled-only tasks (clustering, topic discovery) as a first-class task. These can appear only as operators.
- No online learning in production, no auto-deployment, no UI.
- No domain-specific logic anywhere in `src/`, `claude-plugin/`, or `programs/` (enforced by lint, section 12 T4/T5).
- No LLM calls in the resolver artifact.
- Set-valued families (multi-label, ranking, detection) are **designed for but not in the first release** (section 3).

### 1.5 Principles

1. **The product is a frontier, not a model.** The human picks the operating point.
2. **Domain enters through four slots only:** the data, the loss function ℓ, constraints, optional priors.
3. **The LLM writes code and proposals. It never grades.** Acceptance is decided by a deterministic evaluator.
4. **A certificate is only as valid as its split regime.** Profile and audit exchangeability first.
5. **Earn complexity.** Any added structure must beat "best single operator + same certifier."
6. **Cheapest lever first.** Thresholds before weights before structure before search policy.
7. **Reproducible by construction.** Seeds, content hashes, manifests, locked environments.
8. **Fail loud, label honestly.** Prefer a weaker, truthfully-labelled guarantee to a strong, unsupported one.

---

## 2. Core abstraction

### 2.1 Definitions

- **Unit** `u`: the smallest thing the system can answer or pass on (a row, a field, a span, a series-horizon, a query, an object). It has input `x_u`, gold `y_u` (known on labelled data), optional group `g_u`, optional time `t_u`.
- **Loss** `ℓ(ŷ, y) ∈ [0, 1]`: scores a committed answer against gold. Defaults per family (section 3). Domain knowledge enters here.
- **Commit rule** `C(τ)`: for threshold `τ ∈ [0, 1]` the graph either **RESOLVES** a unit (returns an answer) or **ABSTAINS** (returns a reason). `τ` means "maximum predicted loss at which I will commit." Larger `τ` commits more. `C(τ)` is nested: anything committed at `τ` is committed at every `τ' ≥ τ`.
- **Coverage** `cov(τ) = P(unit committed at τ)`.
- **Selective risk** `R(τ) = E[ℓ | committed at τ]`.
- **Band** `α_j`, `j = 1..m`, with `α_1 < ... < α_m`: ceilings on selective risk. Each band has a policy: `auto` or `audit`.
- **Certified threshold** `τ̂_j`: the largest `τ` on a grid for which `R(τ) ≤ α_j` is certified at level `δ_j = δ / m`.
- **Guarantee (type `pac_high_prob`)**: with probability at least `1 − δ` over the draw of the calibration set, `R(τ̂_j) ≤ α_j` holds for all bands simultaneously, under the stated exchangeability regime.

### 2.2 Tiers

Because `C(τ)` is nested, a unit's tier is the smallest `j` at which it is resolved:

| Tier | Meaning |
|---|---|
| `AUTO` | resolved at a band whose policy is `auto` |
| `AUDIT` | resolved only at a band whose policy is `audit` (auto, plus audit sampling) |
| `RESIDUAL` | unresolved at `τ̂_m`. This is **Set B**, handed to Phase B. |

### 2.3 The commit score (one mechanism for every task)

Each stage `i` of the graph has a **learned acceptability scorer** `ĝ_i(x) ≈ E[ℓ(ŷ_i(x), y) | x, signals_i(x)]`, trained on out-of-fold predictions and calibrated (section 7.7). Its inputs are the uncertainty signals the operator emits (max-prob, margin, interval width, ensemble variance, kNN distance, validators, ...).

- Stage `i` commits unit `x` at `τ` iff `ĝ_i(x) ≤ m_i · τ`, where `m_i` is a per-stage multiplier fixed **before** calibration (default 1.0).
- The answer comes from the **first stage in cascade order** that commits.
- The graph's **commit score** is `s(x) = min_i ĝ_i(x) / m_i`. Then `commit(τ) ⇔ s(x) ≤ τ`. That is monotone in `τ` and nested by construction.

This is a learned-deferral scorer in the spirit of FrugalGPT's learned correctness scorer, generalized from "is the label right" to "is the loss small," so it works for regression, extraction, forecasting and anomaly tasks.

---

## 3. Task families and release sequencing

One harness, many tasks. The table gives defaults. Every default is overridable through the TaskSpec (section 6.1), and overriding triggers the loss validators in section 7.9.

| Family | Commit unit | Default ℓ | Uncertainty signals | Release |
|---|---|---|---|---|
| Classification | row | 0/1 error | max-prob, margin, ensemble variance, kNN agreement | **R1** |
| Regression | row | `1[abs error > τ_tol]` | interval width, quantile spread, ensemble variance | **R1** |
| Forecasting | series × horizon | `1[abs error > τ_tol]` | interval width from backtest residuals | **R1** (weaker guarantee, see 7.5) |
| Extraction / tagging | field or span | `1 − match` (exact or F1) | token/CRF marginals, extractor agreement, validators | **R1** (field-level) |
| Anomaly detection | row (commit as "normal") | missed anomaly among units committed as normal | anomaly score | **R1** |
| Multi-label / set-valued | row | false-negative proportion | per-label probabilities, set size | R2 |
| Ranking / retrieval | query | `1 − recall@k` | score gap, candidate-set size | R2 |
| Detection / segmentation | object or region | `1 − 1[IoU ≥ τ and class right]` | box/mask scores, mask stability | R2 |

Set-valued families need a two-parameter commit rule (inclusion threshold and maximum set size) and CRC/LTT over both. They are designed for (the abstractions already accommodate them) but built after R1 is certified (decision D7).

---

## 4. Architecture and repo layout

### 4.1 Tech stack

- Python 3.11+ (target 3.12), `uv` for environments and lockfile.
- Core: numpy, scipy, pandas/pyarrow (polars optional), scikit-learn, statsmodels, networkx, pydantic v2, typer, rich, joblib, diskcache, sqlite3.
- Optional extras (pyproject extras): `[tabular]` lightgbm/xgboost/catboost; `[autogluon]` AutoGluon tabular/timeseries/multimodal; `[text]` sentence-transformers; `[vision]` torch/torchvision/timm/open_clip; `[ts]` statsforecast; `[search]` openevolve / shinka adapters, tpot; `[dev]` pytest, hypothesis, ruff, mypy, pre-commit.
- No GPU requirement in v1. GPU is an optional budget field.

### 4.2 Repo layout

```
amx/
├─ HANDOFF.md                      # this file
├─ CLAUDE.md                       # durable rules (section 18.3)
├─ README.md
├─ pyproject.toml   uv.lock   Makefile
├─ src/amx/
│  ├─ spec/       # pydantic models, loaders, validators
│  ├─ data/       # adapters, UnitFrame, content-hash cache
│  ├─ profile/    # modality, exchangeability audit, feasibility, fingerprint
│  ├─ split/      # regimes, manifests, vault client
│  ├─ loss/       # builtin + custom losses, validators
│  ├─ ops/        # operator contract, registry, library/
│  ├─ graph/      # typed DAG, constructs, compile, trace
│  ├─ scorer/     # learned acceptability + calibration
│  ├─ cert/       # LTT, CRC, ACI, bounds, n_min, guarantee objects
│  ├─ eval/       # read-only evaluator, OOF runner, fitness, noise margin
│  ├─ search/     # backends: ratchet, evolve, tpot, bandit, random
│  ├─ reasons/    # detectors, analyst tagging
│  ├─ report/     # bands.json/.md, plots, manifest writer
│  ├─ ledger/     # sqlite, fingerprints, warm start
│  ├─ sim/        # synthetic oracle generators, T1 simulators
│  ├─ warden/     # vault, sandboxed resolver runner, freeze tokens
│  └─ cli.py
├─ claude-plugin/
│  ├─ .claude-plugin/plugin.json
│  ├─ skills/amx-*/SKILL.md
│  ├─ agents/{proposer,auditor,analyst,reporter}.md
│  ├─ hooks/hooks.json
│  ├─ hooks/scripts/{guard_paths.py, lint_after_edit.sh}
│  └─ (optional, A4) .mcp.json
├─ programs/                       # program.md search policies per family
├─ datasets/<name>/{spec.yaml, loader.py, NOTES.md}
├─ tests/{unit, property, contract, statistical, e2e, redteam}
├─ docs/{decisions.md, schemas/}
└─ runs/                           # gitignored
```

### 4.3 Dependency direction

`spec → data → profile → split → loss → ops → graph → scorer → cert → eval → search → reasons → report`. `warden` depends on `split`, `cert`, `loss` only. `search` may import `eval` but **never** `cert` internals or `warden`. `cert`, `eval`, `split`, `warden` are the **protected core** (section 8.5).

---

## 5. Run directory, artifacts, CLI

### 5.1 Run directory

```
runs/<run_id>/
├─ taskspec.yaml
├─ profile.json                 # modality, regime, feasibility, fingerprint
├─ splits.manifest.json         # hashes and counts only; no labels
├─ search/
│  ├─ program.md                # the search policy used (human-edited)
│  ├─ graph.yaml                # current graph (agent-editable)
│  ├─ ops_user/                 # agent-authored operators (gated path)
│  └─ ledger.sqlite             # experiments, hypotheses, outcomes
├─ cert/certificate.json        # written only by the warden
├─ report/{bands.json, bands.md, frontier.png}
├─ handoff/{manifest.parquet, taxonomy.md, summary.json, interface.md}
├─ artifact/                    # frozen resolver: graph.yaml, op states, requirements.lock, resolver entrypoint
└─ audit/audit_plan.json
```

Label stores for calibration and sealed folds live **outside** the run tree, in the vault (section 8.4).

### 5.2 Resolver API (the frozen artifact)

```python
import amx
r = amx.load("runs/<id>/artifact")
res = r.resolve(unit)                 # Resolution
res.status        # "RESOLVED" | "ABSTAIN"
res.value         # answer if resolved
res.tier          # "AUTO" | "AUDIT" | None
res.stage         # which stage answered
res.score         # commit score s(x)
res.reason        # reason code if ABSTAIN
res.trace         # stage-by-stage signals and decisions
r.resolve_batch(df)                   # vectorized
r.set_band(j)                         # choose operating point (default: loosest 'auto' band)
```

Deterministic given inputs and artifact. Latency and size budgets are enforced at freeze.

### 5.3 CLI (typer)

```
amx init <dir>                    scaffold run dir from a TaskSpec template
amx doctor                        env check: versions, extras, vault mode, hooks
amx profile -s taskspec.yaml      profile + recommend regime/certifier/feasibility
amx split   -s taskspec.yaml      create hash-locked folds; labels for calib/sealed go to the vault
amx baseline -s taskspec.yaml     baseline zoo: B0 / B1 / B2 (section 12, T2)
amx lint graph.yaml               type-check + operator contract tests + budgets
amx eval -g graph.yaml            READ-ONLY evaluator on dev OOF only; prints metrics JSON
amx search --backend ratchet|evolve|tpot|bandit|random [--supervise]
amx certify --freeze              HUMAN-GATED; spends one certify call (warden)
amx report                        bands.json/.md, frontier plot
amx handoff                       manifest, taxonomy scaffold, summary
amx simulate --t1 [--synthetic|--real]
amx ledger add|query|warmstart
```

`certify`, `report --final`, and `simulate --real` are **warden commands**: they need a freeze token that the agent environment does not have (section 8).

---

## 6. Contracts

### 6.1 TaskSpec (YAML, validated by pydantic)

```yaml
amx_version: 1
name: example-run
data:
  uri: data/train.parquet            # file | dir (images/audio) | loader-defined id
  format: parquet                    # parquet | csv | jsonl | image_folder | custom
  unit_id: id                        # unique unit id column
  inputs:                            # typed; "infer" lets the profiler propose
    - {name: text,   kind: text}
    - {name: amount, kind: numeric}
    - {name: when,   kind: timestamp}
  target: {name: label, kind: categorical}     # categorical | numeric | spans | set | series
  time_column: when                  # optional
  group_columns: []                  # entity / document / session ids that must not straddle folds
  independence_unit: unit            # unit | group:<col>   (see 7.4)
task:
  family: classification             # see section 3
  commit_unit: row
  loss:
    kind: builtin                    # builtin | custom
    name: zero_one                   # zero_one | err_gt_tol | one_minus_f1 | missed_anomaly
    params: {}
    # custom: {kind: custom, path: losses/my_loss.py, fn: loss}
bands:
  alphas:   [0.005, 0.01, 0.02, 0.05]
  policies: [auto,  auto, audit, audit]
  delta: 0.10                        # total certificate failure probability, split across bands
  watch_slices: [{name: by_target, by: target}]
splits:
  regime: auto                       # auto | iid | grouped | temporal | blocked
  fractions: {dev: 0.6, calib: 0.2, sealed: 0.2}
  seed: 1337
  max_certify_calls: 3
budget:
  wall_clock_hours: 4
  cpu_cores: 16
  gpus: 0
  max_experiments: 300
  agent_max_usd: 150                 # enforced by supervisor (--max-budget-usd) and ledger
constraints:
  max_apply_ms_per_unit: 50
  max_model_mb: 500
  allow_pretrained_encoders: true
  allow_gpu: false
  interpretable_only: false
  forbidden_inputs: []
priors:
  label_hierarchy: null
  invariants: []                     # section 10.3
```

Validation rules: bands strictly increasing, in (0, 1); `len(policies) == len(alphas)`; fractions sum to 1; `calib` fraction must support `n_min` for the tightest band (profiler warns, `--force` required to proceed); loss must pass section 7.9.

### 6.2 Operator contract (Python)

```python
from typing import Any, Mapping, Protocol, TypedDict
import numpy as np

class Output(TypedDict):
    value: Any                          # labels | reals | intervals | spans | ...
    signals: Mapping[str, np.ndarray]   # named uncertainty signals, shape (n,)
    provenance: Mapping[str, Any]

class Operator(Protocol):
    name: str
    in_type: str                        # Table | Text | Image | Audio | Series | Doc | Embedding | Features | ...
    out_type: str                       # Label | LabelSet | Real | Interval | Spans | Features | Embedding | ...
    orientation: Mapping[str, str]      # signal -> "uncertainty" | "confidence"
    def fit(self, train: "Fold", ctx: "FitContext") -> None: ...
    def apply(self, x: Any) -> Output: ...
    def params(self) -> Mapping[str, Any]: ...
    def cost(self) -> "Cost": ...       # fit_s, apply_ms_per_unit, size_mb
    def state(self) -> bytes: ...
    @classmethod
    def load(cls, state: bytes) -> "Operator": ...
```

Normative rules:
1. `fit` sees **only** the `train` fold passed in. Never any other fold's labels. The harness provides `ctx.oof()` for stacking.
2. `apply` is pure and deterministic given state and a seed in `ctx`. It must not depend on batch composition (no batch statistics).
3. Every operator emits at least one uncertainty signal, or declares `signals = {}` and is then eligible only as a featurizer/combiner.
4. Operators declare typed ports. The graph compiler rejects type mismatches.
5. `state()` / `load()` round-trip exactly. Graph hash = hash(structure + params + state hashes).
6. Operators must pass the contract tests in section 8.3 before they can be used in a search (`amx lint`).
7. Agent-authored operators live in `ops_user/` and run only in the sandbox (section 8.6).

### 6.3 Graph spec (YAML)

```yaml
version: 1
nodes:
  dedupe: {op: NearDupLookup,       inputs: [text], params: {threshold: 0.98}}
  tfidf:  {op: TfidfFeaturizer,     inputs: [text]}
  lr:     {op: LogisticHead,        inputs: [tfidf]}
  emb:    {op: FrozenTextEncoder,   inputs: [text], params: {model: ENCODER_ID}}
  knn:    {op: KNNHead,             inputs: [emb]}
  ens:    {op: StackEnsemble,       inputs: [lr, knn], params: {meta: logistic}}
  guard:  {op: Verifier,            inputs: [ens], params: {invariants: []}}
resolve:
  cascade:
    - {stage: dedupe, multiplier: 1.0}
    - {stage: guard,  multiplier: 1.0}
loops: []          # e.g. [{id: st, body: ens, stop: {max_iter: 3, min_gain: 0.001}}]
```

Constructs: `Seq`, `Ensemble(mean|vote|stack|bayes)`, `Mixture/Gate`, `Router(by=features|uncertainty)`, `Cascade`, `Loop(body, stop, max_iter)`, `Verifier(invariants)`. Loops must have a hard iteration cap and are unrolled for latency accounting. The compiler computes worst-case latency and size and rejects graphs over budget.

### 6.4 Evaluator output (`amx eval`, what the agent sees)

The agent sees **dev OOF metrics only**. It never sees calibration or sealed metrics.

```json
{
  "graph_hash": "sha256:...",
  "dev_oof": {
    "bands": [
      {"alpha": 0.01, "tau_hat": 0.0123, "coverage_lcb": 0.412,
       "risk_est": 0.0087, "n_committed": 8123}
    ],
    "aurc": 0.0123,
    "worst_slice_gap": 0.041,
    "stage_attribution": {"dedupe": 0.12, "ens": 0.31}
  },
  "cost": {"fit_s": 412.0, "apply_ms_p50": 1.9, "apply_ms_p99": 7.4, "size_mb": 38.2},
  "n_nodes": 7,
  "fitness": 0.5123,
  "notes": []
}
```

### 6.5 Certificate and report (`bands.json`)

```json
{
  "run_id": "...", "amx_version": "...",
  "taskspec_hash": "...", "graph_hash": "...",
  "guarantee": {
    "type": "pac_high_prob",
    "delta": 0.10, "delta_per_band": 0.025,
    "regime": "iid",
    "assumptions": ["calib and future units exchangeable under regime=iid"],
    "certify_calls_used": 1, "certify_calls_max": 3,
    "calib_n": 12000, "sealed_n": 12000
  },
  "bands": [
    {"alpha": 0.01, "policy": "auto", "tau_hat": 0.0123,
     "thresholds": {"dedupe": 0.0123, "ens": 0.0123},
     "coverage_calib": {"est": 0.41, "ci95": [0.40, 0.42]},
     "coverage_sealed": {"est": 0.41, "ci95": [0.40, 0.42]},
     "risk_sealed": {"est": 0.0088, "cp_upper95": 0.0102},
     "n_committed_calib": 4920, "n_min_required": 368,
     "per_slice": [{"name": "class=A", "n": 210, "risk": 0.004, "cp_upper95": 0.014, "flag": null}],
     "stage_attribution": {"dedupe": 0.12, "ens": 0.29}}
  ],
  "frontier": [{"tau": 0.005, "coverage": 0.30, "risk_ucb": 0.006}],
  "warnings": [],
  "costs": {"latency_ms_p50": 1.9, "latency_ms_p99": 7.4, "model_mb": 38.2}
}
```

`guarantee.type` is one of `pac_high_prob`, `expectation`, `long_run_frequency`, `holdout_empirical`, `none` (section 7.1). `warnings` must include any n_min shortfall, any regime downgrade, and any slice whose upper bound exceeds `2 × alpha`.

### 6.6 Handoff manifest (Phase A to Phase B)

`handoff/manifest.parquet`, one row per unit in `calib ∪ sealed ∪ (optional new data)`:

| Column | Meaning |
|---|---|
| `unit_id`, `split` | identity and fold |
| `tier` | `AUTO` / `AUDIT` / `RESIDUAL` |
| `stage_resolved`, `value`, `commit_score` | if resolved |
| `reason_primary`, `reason_flags` | section 10.1 |
| `candidates` | top-k outputs with scores, or the set/interval |
| `cluster_id`, `difficulty_rank` | residual structure |
| `epistemic`, `aleatoric`, `ood_score`, `support_n` | decomposed uncertainty and support |
| `verifier_failures` | invariants violated |
| `gold` | if known (never exposed for sealed to agents) |

Plus `taxonomy.md` (failure modes, frequency, example ids, recommended handling as *hypotheses*), `summary.json` (counts per tier, reason, cluster), `interface.md` (how Phase B reads it). Important note for Phase B: **an LLM's accuracy on Set B is not its overall accuracy.** Set B is selected by Phase A as the hard tail.

### 6.7 Ledger (SQLite)

```sql
runs(run_id, created, taskspec_hash, fingerprint_json, backend, status)
experiments(exp_id, run_id, parent_exp, graph_hash, hypothesis, edit_json,
            fitness, metrics_json, cost_json, accepted, created)
fingerprints(run_id, n_rows, n_features, modality, family, n_classes, imbalance,
             noise_est, text_len_p50, time_structure, group_structure, ...)
winners(run_id, graph_hash, certified_coverage_json, guarantee_type)
```

The agent may `INSERT` hypotheses and read experiments. Fitness values are written only by `amx eval`.

---
## 7. Statistics (normative)

### 7.1 Guarantee types and certifier selection

The profiler (`amx profile`) selects the certifier. The user does not.

| Situation | Certifier | `guarantee.type` |
|---|---|---|
| Selective prediction, binary or `[0,1]` ℓ, regime `iid` or `grouped` | Learn-then-Test (LTT), fixed-sequence over a τ grid. Exact binomial p-values (binary ℓ) or Hoeffding–Bentkus p-values (fractional ℓ). | `pac_high_prob` |
| Set-valued, monotone ℓ (R2 families) | Conformal Risk Control (expectation) or LTT (high probability) over the inclusion threshold | `expectation` / `pac_high_prob` |
| Interval forecasts under temporal dependence | Adaptive Conformal Inference (ACI), per horizon | `long_run_frequency` (interval miscoverage only) |
| Temporal / blocked regime with selective commit | LTT on a calibration **window** after dev, plus rolling re-certification | `holdout_empirical` (valid only if calibration-to-future is stationary) |
| Profiler cannot establish any valid regime | Descriptive numbers only | `none` |

Why LTT is the default: selective risk is a ratio of expectations and is **not monotone** in the threshold. CRC requires a monotone loss, and its finite-sample guarantee fails when the loss is not monotone in the threshold. LTT drops both the one-dimensional-threshold and the monotonicity assumptions and lists selective classification and selective regression among its applications.

### 7.2 Certification procedure (fixed-sequence LTT along τ)

1. **Grid.** `τ_1 < ... < τ_G`, conservative to liberal. Default `G = 200`, log-spaced from `1e-4` to `1`. Fix the grid before touching calibration data.
2. **Per-unit quantities.** For each calibration unit `u`: commit score `s_u`, and for each `τ_g` the loss `ℓ_u(τ_g)` of the answer the graph would give (answer from the first stage whose scaled score is `≤ τ_g`).
3. **Order and start point, fixed on dev before calibration.** Fixed-sequence testing needs an order chosen in advance, and dev data may be used to choose it. Starting at the most conservative grid value would stop immediately, because almost nothing is committed there and the p-value cannot reject. So, per band `j`:
   - `need_j = n_min(α_j, δ_j)` (section 7.4).
   - `start_j` = the smallest grid index whose dev-estimated committed count on the calibration set, `cov_dev(τ) · n_calib`, is at least `1.25 · need_j`.
   - If no such index exists, the band is **infeasible on dev**. Say so at profile time; do not run it.
4. **Walk.** Test `H_g : R(τ_g) > α_j` for `g = start_j, start_j + 1, ...` (liberal-ward):
   - `n_g = #{u : s_u ≤ τ_g}`. If `n_g < need_j`: stop with status `sample_size_limited`.
   - `R̂_g` = mean of `ℓ_u(τ_g)` over committed units; `p_g` = p-value (section 7.3).
   - If `p_g ≤ δ_j`: certify `τ_g` and continue. Otherwise stop with status `risk_limited`.
5. `τ̂_j` = last certified `τ_g`. If none: band uncertifiable, coverage 0, warning.
6. **Monotonize across bands:** `τ̂'_j = max_{k ≤ j} τ̂_k`. This is valid because a threshold certified for a tighter band also satisfies every looser band; failure probabilities already add up through the `δ / m` budget. After this step `τ̂'_1 ≤ ... ≤ τ̂'_m` holds by construction. Assert it in code and in a property test.
7. **Budget `δ` across bands** by Bonferroni (`δ_j = δ / m`, cheap for `m ≤ 5`) and across certify calls (section 7.11).

Fixed-sequence testing controls the family-wise error at `δ_j` per band with arbitrary dependence between the tests. The commit decision must be a deterministic function of `x` and of parameters fixed before calibration, so the committed units' losses are i.i.d. draws from the conditional distribution given "committed." The warden enforces this by running a frozen graph.

**Fallback mode `--cert-mode bonferroni`:** Bonferroni over the entire grid (`δ_j / G` per test). It needs no ordering and is valid, but needs roughly 2 to 3 times the committed calibration units (for example about 895 instead of 368 at `α = 1%`, `δ_j = 0.025`, `G = 200`). Use it as a diagnostic cross-check, not as the default.

### 7.3 p-values (VERIFY and Monte Carlo test)

- **Binary ℓ.** `k` = number of losses among `n` committed units. `p = P(Binomial(n, α) ≤ k)` (scipy `binom.cdf`).
- **Fractional ℓ in [0,1]** (Hoeffding–Bentkus form used in LTT/RCPS):
  `p = min( exp(-n · h1(min(R̂, α), α)),  e · P(Binomial(n, α) ≤ ceil(n · R̂)) )`,
  with `h1(a, b) = a ln(a/b) + (1-a) ln((1-a)/(1-b))`.
  **VERIFY** against the LTT/RCPS papers before use. Add a Monte Carlo validity test with Beta-distributed losses (the empirical rejection rate under the null boundary must be at most the nominal level).
- Handle edge cases: `R̂ = 0`, `R̂ ≥ α` (p = 1), `n = 0` (p = 1).

### 7.4 Minimum sample sizes (the feasibility check)

With zero observed losses, certifying `R ≤ α` at level `δ_j` needs

`n_min(α, δ_j) = ceil( ln(δ_j) / ln(1 − α) )` **committed calibration units** (more if any losses are observed).

| α \ δ_j | 0.100 | 0.025 |
|---|---|---|
| 0.5% | 460 | 736 |
| 1% | 230 | 368 |
| 2% | 114 | 183 |
| 5% | 45 | 72 |

`amx profile` estimates coverage at each band from dev OOF and computes the expected number of committed calibration units. If it falls below `n_min` it recommends: a larger calibration fraction, looser bands, or fewer bands (smaller `m` raises `δ_j`). It then requires `--force` to proceed.

Rule of thumb: per-class (class-conditional) guarantees at tight bands are usually infeasible. Certify marginally; monitor slices (7.10).

### 7.5 Independence unit, clustering, regimes, shift

- **Independence unit.** If `independence_unit` is `group:<col>` (fields in a document, horizons of a series, repeated entities), certify at group level. For each group with at least one committed unit, `ℓ_g` = mean loss over its committed units. The certified selective risk is the mean of `ℓ_g` over groups (groups equally weighted) and `n` counts groups. Report both group-weighted and unit-weighted empirical risks.
- **Regimes** (chosen by the profiler's exchangeability audit, section 8.2): `iid` (stratified random), `grouped` (no group straddles folds), `temporal` (dev < calib < sealed in time, with an embargo of at least the maximum horizon or lag), `blocked` (spatial or block CV).
- **Forecasting.**
  - Intervals: split-conformal per horizon on the calibration window, then ACI online: `α_{t+1} = α_t + γ (α_target − err_t)`, default `γ = 0.005`; track local coverage over a 300-step window.
  - Commit rule: interval half-width `≤ τ_tol`; `ℓ = 1[abs(error) > τ_tol]`.
  - **Caveat (decision D15).** ACI gives long-run *marginal* interval coverage. Selecting by interval width breaks conditional validity, so the selective risk is reported as empirical with block-bootstrap intervals and labelled `holdout_empirical`. Classical split conformal can lose validity on time series. Do not present forecasting results with a stronger label.
- **Shift.** No distribution-free guarantee survives arbitrary shift. Ship the audit plan (7.12) and a drift alarm instead.

### 7.6 What is never claimed

Per-class or per-slice risk bounds (unless 7.10 enables and satisfies them), performance under shift, and any statement about Phase B outcomes. Every report prints this list.

### 7.7 Learned scorer

- Target: per-unit loss `ℓ` from dev **OOF** predictions of the stage.
- Features: the operator's uncertainty signals (and optionally cheap meta features such as input length).
- Model: monotone-constrained GBDT or logistic / isotonic regression. Calibrate with isotonic when `n_oof ≥ 2000`, otherwise Venn–Abers or temperature scaling.
- Cascade subtlety: stage `i` only ever sees units that earlier stages did not commit. Fit scorer `i` on all OOF units, then **recalibrate on the units that reach stage `i` at a reference `τ_0`** (default `τ_0 = α_m`).
- If the scorer carries no signal (cross-validated AUROC near 0.5), the stage commits nothing and flags `irreducible`.
- The scorer is part of the graph and is trained on dev only.

### 7.8 Search-time statistics

- **Fitness input.** On dev OOF, run a lenient LTT-lite (`δ_search = 0.2`, Bonferroni across bands) to get `coverage_lcb_j`. This estimates, never certifies.
- **Acceptance margin.** Paired bootstrap over dev units (`B = 200`) of the fitness difference. Accept a candidate iff `ΔF > max(ε, z · SE_paired)`; initial `ε = 0.002`, `z = 1.0`.
- **Adaptive overfitting.** Structure search has more degrees of freedom than hyperparameter search. After `N_max = 150` cumulative accepted comparisons on the same dev data, switch the evaluator to Thresholdout-style noisy feedback (Laplace noise on reported fitness) or refresh with a held-back "search reserve" fold. Record the switch.
- Calibration is touched only by the warden, only at freeze.

### 7.9 Loss validation (at spec load and on any override)

- Range: `ℓ ∈ [0,1]`; `ℓ(y, y) = 0`; check on gold-vs-gold and random pairs.
- Monotonicity (set-valued families only): `ℓ` non-increasing as the inclusion threshold grows. Selective families need no monotonicity (LTT).
- Loss-distribution report: histogram of `ℓ` on baseline predictions. Warn if a fractional loss has more than 95% mass at `{0, 1}`, or if a tolerance `τ_tol` makes `ℓ` nearly constant.
- Custom losses or overridden family defaults require `--confirm-loss` (human).
- Unbounded losses may be squashed by a monotone map (for example `arctan`), but then the band is in squashed units. State that in the report.

### 7.10 Slices and disparities

- Report per-slice empirical selective risk with Clopper–Pearson upper bounds. Claim no guarantee unless `n_committed ≥ n_min` for that slice **and** slice-conditional (Mondrian) calibration is enabled (default off).
- Selective classification can improve average accuracy while widening gaps between groups, and can even lower accuracy for some groups; it behaves well on models that are already similar across groups at full coverage. Therefore fitness includes `worst_slice_gap`, and every report includes per-slice selective risk.
- Flag any slice whose upper bound exceeds `2 × α_j`.

### 7.11 Certify-call budget and the sealed fold

- `max_certify_calls` (default 3). Each call uses `δ / max_certify_calls`. The count lives in an append-only file in the vault; the warden refuses further calls.
- The sealed fold is **single-touch per release**. A second `report --final` needs fresh sealed data (`--new-sealed`) or is rejected.

### 7.12 Audit plan (shipped with every artifact)

`audit/audit_plan.json`: a random sample of `AUTO` and `AUDIT` resolutions is routed to ground truth continuously (initial rate: enough to estimate live risk to within `±α_1` at 95%, minimum 1%). This provides an unbiased live-risk estimate and a drift alarm. The certificate holds only if future units come from the audited regime.

---

## 8. Splits, leakage, and sealed-fold protection

### 8.1 Fold roles

| Fold | Default | Who may see it | Purpose |
|---|---|---|---|
| `dev` (= train ∪ search) | 60% | agent: inputs and labels | fit operators; search via K-fold OOF (default K = 5) |
| `calib` | 20% | **warden only** (inputs and labels) | certification, budgeted calls |
| `sealed` | 20% | **warden only** | final report, single touch |

Initial defaults. Override for small data. If `n < 3000`, warn and require `--allow-small` (cross-fitted calibration is an A4+ enhancement).

### 8.2 Splitting rules and the exchangeability audit

The profiler checks and reports:
- **Time**: timestamp-typed or monotone columns, and columns named like dates. Recommend `temporal`.
- **Groups**: repeated high-cardinality IDs (entities, documents, sessions). Recommend `grouped` and set `group_columns`.
- **Duplicates and near-duplicates**: exact row hashes; MinHash for text; embedding cosine `> 0.98`; perceptual hash for images. Near-duplicate clusters are assigned to **one fold**. Any calibration or sealed unit that matches a dev unit is dropped from calibration/sealed (and counted in the report).
- **Target quirks**: pile-up at a cap (censoring), extreme imbalance, label noise estimate.
- **Feasibility**: `n_min` check (7.4).

Splits are hash-locked: `splits.manifest.json` stores per-fold id hashes and counts only. Stratification follows the target for classification and quantile bins for regression.

### 8.3 Operator contract tests (run by `amx lint`)

1. **Label-shuffle invariance.** Shuffle labels of every non-train fold; `apply` outputs on those folds' inputs must be unchanged.
2. **Fit-only-on-train.** Replace non-train labels by NaN; the fitted state hash must be identical.
3. **Batch invariance.** `apply` on single units equals `apply` on the batch.
4. **Determinism.** Same seed and state give the same output.
5. **Serialization round-trip.** `load(state())` reproduces outputs exactly.
6. **Causality** (temporal operators). Truncating data after time `t` leaves outputs at `t` unchanged.
7. **Budgets.** `apply_ms_per_unit` and `size_mb` within constraints.

### 8.4 Vault and warden

The **vault** stores calibration and sealed inputs and labels. The **warden** is the only process that touches them.

Flow at freeze (`amx certify --freeze`, human-gated):
1. The warden loads the frozen graph artifact and starts a **sandboxed resolver process** that receives **inputs only** (no labels) and returns outputs.
2. The warden computes losses against the labels itself, runs section 7.2, writes `cert/certificate.json`, and increments the certify counter.
3. Only aggregate results and the handoff manifest columns allowed for calib/sealed leave the warden. `gold` for sealed is never written to agent-visible paths.

Vault modes (`amx doctor` reports which is active):

| Mode | Isolation | Use |
|---|---|---|
| `local_dir` | labels and inputs in `~/.amx/vault/<run_id>/`, mode 0700, ideally owned by a separate OS user | convenience only. **Not safe** against a determined agent running as the same user. |
| `container` (**recommended**) | agent container mounts only `runs/<id>` (dev); the warden runs in a sibling container holding the vault; a narrow CLI/HTTP API exposes `certify`, `final_report`, `simulate` with a freeze token | default for real work |
| `remote` | warden on a separate host or account | high assurance |

The freeze token is issued by the human or by the orchestrator script after the search budget completes. It is never in the agent's environment.

### 8.5 Protected core

During search sessions agents may **not** edit: `src/amx/{cert,eval,split,warden,loss}/**`, `tests/{statistical,redteam}/**`, `claude-plugin/hooks/**`, `.claude/settings*`. Enforce with a PreToolUse hook, permission deny rules, and branch protection / CODEOWNERS. Changes there need a decision-log entry and owner approval.

### 8.6 Sandbox for agent-authored operators

Subprocess with CPU, memory and wall-clock limits; no network; read-only mount of dev data; no access to vault paths; import allowlist; best-effort static scan (`os.environ`, `subprocess`, sockets, `open` of non-allowlisted paths). The sandbox is the protection. The scan is only a tripwire.

### 8.7 Defense in depth for Claude Code sessions

1. **Isolation** (container or OS user) is the primary control.
2. **Permission deny rules** for Read/Edit/Write/Bash patterns on vault and protected paths. Deny rules override allow pre-approvals.
3. **PreToolUse hook** guard (exit code **2** blocks; exit 1 does not; matchers are case-sensitive).
4. **Auditor subagent** red-teams the guards each milestone (attempts to read the vault must fail).
5. Limits to remember: hooks are a policy layer that only covers what you anticipated; files referenced with `@` in a prompt skip PreToolUse, so pair hooks with Read deny rules; `claude -p --bare` skips hooks, skills, plugins, MCP and CLAUDE.md auto-discovery, so the supervisor must pass explicit settings if it uses `--bare`.

---

## 9. Search

### 9.1 Backends (one interface, one evaluator, one budget)

```python
class SearchBackend(Protocol):
    def propose(self, state: SearchState) -> list[Proposal]: ...
    def observe(self, results: list[EvalResult]) -> None: ...
```

| Backend | What it is | Role |
|---|---|---|
| `ratchet` | Claude Code in the loop, autoresearch-style: one mutable spec file, fixed evaluator, keep/discard by rule, git as memory | primary agentic backend |
| `evolve` | adapter to OpenEvolve or ShinkaEvolve: an `evaluate.py` wrapping `amx eval` (returns a dict of scalars), seed graph, LLM mutations (Claude via headless CLI), island/archive/novelty machinery for free | second agentic backend |
| `tpot` | TPOT-style genetic programming over DAG pipelines on supported families (classification, regression) | classical structure-search baseline |
| `bandit` | Thompson sampling over a fixed template library plus Optuna HPO | cheap baseline |
| `random` | random edits from the same grammar | sanity floor |

Do not write an evolutionary controller from scratch. Write the evaluator and the grammar.

### 9.2 Edit space

`edit_json` against `graph.yaml`: add/replace/remove/rewire nodes, change params, add/remove/reorder cascade stages, change stage multipliers, add a loop, add a verifier. New operators go in `ops_user/` through the gated path (`amx lint` must pass; sandbox only).

### 9.3 Fitness

```
F(G) =  Σ_j w_j · coverage_lcb_j
      − λ_lat   · log1p(apply_ms_p50)
      − λ_size  · log1p(size_mb)
      − λ_gap   · worst_slice_gap
      − λ_nodes · n_nodes / 10
```

Initial: `w_j = 1/m`, `λ_lat = 0.002`, `λ_size = 0.001`, `λ_gap = 0.1`, `λ_nodes = 0.002`. Record tuning in the decision log. Latency and size are in the fitness so a transformer must beat a logistic regression on certified coverage per unit cost.

### 9.4 Acceptance rule

Accept iff: graph passes `amx lint` and budgets, and `ΔF > max(ε, z · SE_paired)` (7.8). On accept: commit on the search branch with the hypothesis in the message. On reject: revert. Every proposal records a falsifiable hypothesis and its outcome in the ledger.

### 9.5 Long runs: external supervisor, not a single session

`amx search --supervise` is a Python loop that launches bounded Claude Code batches:

```bash
timeout 3600 claude -p "$(cat programs/ratchet_batch_prompt.md)" \
  --plugin-dir ./claude-plugin \
  --output-format json \
  --max-turns 60 --max-budget-usd 10 \
  --allowedTools "Read,Grep,Glob,Edit,Bash(amx eval *),Bash(amx lint *),Bash(amx ledger *)" \
  --permission-mode dontAsk \
  > batches/batch_001.json
```

(VERIFY flags against current docs.) The JSON result carries `result`, `session_id` and `total_cost_usd`; log them to the ledger. Continue with `--resume <session_id>` or start fresh with ledger context each batch. Stop on budget, wall-clock, plateau (no accepted edit in 25 proposals), or token cap.

Why not a Stop-hook loop alone: runaway Stop hooks are capped (documented at eight consecutive blocks from roughly v2.1.143). Do not make correctness depend on a single session staying alive.

### 9.6 Parallelism

Up to 4 proposer subagents with `isolation: worktree` initially; evaluator jobs on a CPU pool. The supervisor merges: best accepted candidate per round, or all non-conflicting edits. Prefer less parallelism: every subagent re-reads CLAUDE.md and its skills.

### 9.7 `program.md` (the human-edited search policy)

Per family template in `programs/`: objective, allowed files, forbidden files, edit grammar, hypothesis format, acceptance rule, failure handling, stop rules, the "never stop to ask" clause for batch sessions, and the "never touch vault/protected core" clause. This file is the main lever on agent behavior; keep it short and exact.

### 9.8 A/B procedure (T2)

Equal experiment budgets (initial: 150), 5 seeds per backend. Compare certified coverage on sealed (via warden) and the OOF fitness trajectory. Paired bootstrap across seeds. Report experiments-to-90%-of-final.

### 9.9 Warm start (A4)

Fingerprint (rows, features, modality, family, class cardinality, imbalance, noise estimate, text length, time and group structure) to nearest ledger datasets to seed graphs and operator-family priors. Evaluate leave-one-**domain**-out (T5). Same idea as Auto-sklearn-style meta-learning plus MLZero-style episodic memory.

---

## 10. Reason codes, verifiers, analyst, handoff to Phase B

### 10.1 Reason codes (task-agnostic)

Assigned to every `RESIDUAL` unit. Keep all flags; pick a primary by priority (top row wins).

| Code | Detector (initial) | Ground truth for T3 | Phase B hypothesis (to be validated, not assumed) |
|---|---|---|---|
| `constraint_violation` | any verifier fired | injected violations | repair or human; LLM only as a cross-check |
| `novel` | OOD score > q99 of in-distribution calibration scores (kNN distance in embedding/feature space, k = 10; isolation forest for tabular) | CLINC150 out-of-scope | LLM zero-shot may help; flag possible taxonomy gap |
| `sparse_support` | nearest class/region has `< 30` training units | rare classes (Covertype minority) | LLM few-shot with retrieved exemplars may help |
| `noisy_gold` | diverse-ensemble consensus (≥ 80% members, mean confidence ≥ 0.9) disagrees with gold; OOF self-confidence of gold ≤ 0.1 | injected label flips | audit labels before any LLM scoring; do not send to LLM |
| `disagreement` | diverse-ensemble mutual information > q90 (epistemic) | synthetic dataset-shift sets | more data or ensembling; LLM tie-breaker maybe |
| `ambiguous` | calibrated candidate set of size 2 to 3 with stable membership; for regression: wide but members agree | CIFAR-10H human-label entropy | LLM selection from the candidate set with a rubric |
| `irreducible` | predicted distribution ≈ prior (KL below threshold); interval width ≈ marginal width; scorer without signal | Adult (expected dominant) | **do not send to an LLM**; accept risk, human, or collect features |

Priority order: `constraint_violation > novel > sparse_support > noisy_gold > disagreement > ambiguous > irreducible`.

### 10.2 T3 targets (initial, revisable via decision log)

AUROC ≥ 0.80 for `novel` on CLINC150 OOS; AUROC ≥ 0.75 for `ambiguous` against top-decile CIFAR-10H entropy; recall ≥ 0.90 for `sparse_support` on minority classes; precision@k ≥ 0.70 and recall ≥ 0.50 for `noisy_gold` with 5% injected flips; precision and recall ≥ 0.95 for `constraint_violation` on injected violations.

### 10.3 Verifiers and invariants

Kinds: range, type/schema, regex/format, cross-field relation (`a ≤ b`, sum equality), monotone relation, unit/scale consistency, uniqueness, lookup/referential. Sources:
1. User priors in the TaskSpec.
2. **Candidates proposed by the profiler or analyst**. Adopt only if they hold on ≥ 99.9% of train units, are stable across folds, and are validated on dev.

A Verifier node abstains with `constraint_violation` on violation (or applies a declared repair operator). Verifiers are a free, domain-agnostic abstention trigger.

### 10.4 Analyst procedure (design-time Claude, read-only)

Input: a harness-produced sample file (≤ 200 residual units, stratified by primary reason and cluster) with signals and nearest training neighbours. **No vault access.**
Output: `handoff/taxonomy.md` with failure-mode ids, definitions, **deterministic tagging rules** (regex, embedding-cluster, feature predicates), example ids, and `handling_hypothesis ∈ {llm, human, none, collect_data}` with confidence `{low, med, high}`. The harness then tags all residual units with the rules and counts units per failure mode. The analyst may not change thresholds, graphs or the certificate.

### 10.5 Phase B interface contract (manifest v1)

- Phase B reads `manifest.parquet`, `taxonomy.md`, `summary.json` plus the raw data.
- Phase B must evaluate on `RESIDUAL` units, stratified by reason and cluster, using a fixed human-audited sample for truth. Overall LLM accuracy is irrelevant.
- `manifest_version: 1` is frozen after milestone A3 so Phase B can start in parallel.

---

## 11. Claude Code plugin design

Facts below were checked against Claude Code documentation around October 2026 (versions around 2.1.28x). Everything here is **VERIFY** at build time: this surface changes fast. Use the official docs and `claude plugin validate`.

### 11.1 Layout and manifest

A plugin is a directory with a manifest at `.claude-plugin/plugin.json` and components at the root: `skills/`, `agents/`, `hooks/`, optional `.mcp.json`, optional `commands/` (custom commands are merged into skills).

```json
{
  "name": "amx",
  "version": "0.1.0",
  "description": "AutoMaX: design and certify LLM-free pipelines with risk-coverage guarantees",
  "author": "TBD",
  "license": "TBD"
}
```

Plugin skills are namespaced (for example `/amx:amx-search`). Keep each `SKILL.md` under about 500 lines; body content stays in context after load, so every line is a recurring token cost.

### 11.2 Skills

| Skill | Invocation | Purpose |
|---|---|---|
| `amx-init` | user | scaffold run dir; author TaskSpec with the owner; require loss confirmation |
| `amx-profile` | user or model | run profiler; summarize regime, feasibility; propose TaskSpec edits |
| `amx-baseline` | model | build baseline graphs; run B0/B1/B2 |
| `amx-search` | **user only** (`disable-model-invocation: true`) | bounded ratchet session per `program.md`; never certifies |
| `amx-operator-author` | model | how to write an operator and pass contract tests |
| `amx-reasons` | model | run detectors; analyst procedure |
| `amx-report` | model | assemble `bands.md` from JSON; no new numbers |
| `amx-certify` | **user only** | documents the human step `amx certify --freeze` in a separate shell; the skill itself must not run it |
| `amx-ledger` | model | log hypotheses and outcomes; warm start |

Frontmatter example (fields per current docs; `allowed-tools` is **pre-approval, not a restriction**, so pair with deny rules):

```markdown
---
name: amx-search
description: Run a bounded AutoMaX architecture-search session on the current run directory. Use when asked to search, improve the graph, or run N experiments. Never certifies.
disable-model-invocation: true
allowed-tools: Read Grep Glob Edit Bash(amx eval *) Bash(amx lint *) Bash(amx ledger *)
---
```

### 11.3 Subagents

| Agent | Tools | Notes |
|---|---|---|
| `proposer` | Read, Grep, Glob, Edit, Bash | `model: sonnet`, `effort: medium`, `maxTurns: 25`, `isolation: worktree`, `skills: [amx-search, amx-operator-author, amx-ledger]`. Edits limited by hook to `graph.yaml`, `ops_user/`, notes. |
| `auditor` | Read, Grep, Glob, Bash(`amx lint`, `amx doctor`) | strongest model. Leakage hunt in `ops_user/` code; red-teams the guards (attempts to read the vault must fail); checks report claims against files. Read-only. |
| `analyst` | Read, Grep, Glob | reads only the harness sample file; writes only `handoff/taxonomy.md` |
| `reporter` | Read, Edit(`report/**`) | renders from JSON; invents no numbers |

Example frontmatter:

```markdown
---
name: proposer
description: Proposes one graph edit per turn with a falsifiable hypothesis, then evaluates it with `amx eval`. Use proactively inside amx-search.
tools: Read, Grep, Glob, Edit, Bash
model: sonnet
maxTurns: 25
isolation: worktree
skills: [amx-search, amx-operator-author, amx-ledger]
---
```

Gotchas to design around:
- Plugin-distributed agents **do not apply** `hooks`, `mcpServers` or `permissionMode` from frontmatter. Put guards in plugin-level `hooks/hooks.json` and in project `.claude/settings.json`, or copy the agent files into `.claude/agents/`.
- Recent versions run subagents in the **background by default with a reduced built-in tool set** (Read, Grep, Glob, Bash, Edit, Write, WebFetch, WebSearch, plus MCP tools). If an agent works in the foreground but not in the background, suspect this.
- Subagents do **not inherit skills**; list them in `skills`.
- A subagent cannot be stricter than a session running in `bypassPermissions` or `acceptEdits`. Never run supervised sessions in those modes with a vault on the same machine.

### 11.4 Hooks (deterministic layer)

Hooks fire before permission checks, so a hook `deny` holds even in permissive modes, but a hook `allow` cannot override a deny rule. Exit **2** blocks on events that can block (PreToolUse blocks the tool call; Stop keeps Claude working); exit 1 is a non-blocking error.

| Event | Matcher | Action |
|---|---|---|
| `PreToolUse` | `Read\|Edit\|Write\|MultiEdit\|NotebookEdit\|Grep\|Glob\|Bash` | `guard_paths.py`: block vault, protected core, label columns of non-dev folds, `amx certify`, `amx report --final`, `amx simulate --real`, `sudo`, network fetchers |
| `PostToolUse` | `Edit\|Write` | `lint_after_edit.sh`: run `amx lint` when `graph.yaml` or `ops_user/` changed; surface stderr as feedback (cannot undo the edit) |
| `SessionStart` | none | print run status: budget left, experiments done, best fitness |
| `Stop` | none | optional: refuse to stop while budget remains and no plateau; capped, so the supervisor owns long runs |

Illustrative `hooks/hooks.json` (VERIFY whether the file expects a top-level `hooks` key; run `claude plugin validate`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Edit|Write|MultiEdit|NotebookEdit|Grep|Glob|Bash",
        "hooks": [
          { "type": "command",
            "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/guard_paths.py",
            "timeout": 10 }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/lint_after_edit.sh",
            "timeout": 120 }
        ]
      }
    ]
  }
}
```

Illustrative guard (regexes are a tripwire; **isolation is the control**):

```python
#!/usr/bin/env python3
"""PreToolUse guard. Reads the event JSON on stdin. Exit 2 blocks and feeds stderr to Claude."""
import json, re, sys

FORBIDDEN_ANY = [r"/\.amx/vault/", r"(^|/)vault/", r"folds/(calib|sealed)", r"AMX_FREEZE_TOKEN", r"(^|/)\.env\b"]
PROTECTED_WRITE = [r"src/amx/(cert|eval|split|warden|loss)/", r"tests/(statistical|redteam)/",
                   r"claude-plugin/hooks/", r"\.claude/settings"]
FORBIDDEN_CMDS = [r"\bamx\s+certify\b", r"\bamx\s+report\b.*--final", r"\bamx\s+simulate\b.*--real",
                  r"\bsudo\b", r"\b(curl|wget)\b"]

def block(msg):
    print(f"BLOCKED by amx guard: {msg}", file=sys.stderr)
    sys.exit(2)

ev = json.load(sys.stdin)
tool = ev.get("tool_name", "")
ti = ev.get("tool_input") or {}
paths = [str(ti[k]) for k in ("file_path", "path", "notebook_path") if k in ti]
cmd = str(ti.get("command", ""))
blob = " ".join(paths + [cmd, str(ti.get("pattern", ""))])

for pat in FORBIDDEN_ANY:
    if re.search(pat, blob):
        block(f"forbidden path or token ({pat})")
if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
    for pat in PROTECTED_WRITE:
        if any(re.search(pat, p) for p in paths):
            block(f"protected file ({pat})")
if tool == "Bash":
    for pat in FORBIDDEN_CMDS:
        if re.search(pat, cmd):
            block(f"forbidden command ({pat})")
sys.exit(0)
```

Test hooks by piping sample event JSON to the script and checking `$?`. Run `claude --debug` to inspect hook resolution.

### 11.5 Settings (project `.claude/settings.json`)

Add **permission deny rules** for the vault and protected paths (Read/Edit/Write/Bash patterns). Deny and ask rules override `allowed-tools` pre-approvals. VERIFY the rule syntax in the permissions docs. Commit the file; enforce with branch protection.

### 11.6 Dev loop for the plugin

```
claude --plugin-dir ./claude-plugin      # load from disk for the session
/reload-plugins                          # after edits
claude plugin validate ./claude-plugin   # validates manifest, frontmatter, hooks
```

### 11.7 MCP

Not in v1. CLI plus JSON files is simpler and auditable. Consider an MCP server in A4 only if tool-call ergonomics demand it.

---
## 12. Acceptance tests T1 to T5

All thresholds are **initial**; tune only through the decision log.

### T1. The certificate holds

**T1-synth (exact, runs in CI).**
- Generators in `amx.sim`: (a) K-class Gaussian mixture with known posterior and a plug-in classifier with controllable miscalibration; (b) heteroscedastic regression with known noise function; (c) AR(1)+seasonal series with a regime shift; (d) clustered units (groups with within-group correlation).
- Procedure: fix the graph and scorer fitted on synthetic dev. Compute population risk `R(τ)` exactly or from at least 1e6 fresh samples. Draw `R = 2000` calibration sets for each `n_calib ∈ {500, 2000, 10000}`. Run the certifier. A violation is `R(τ̂_j) > α_j`.
- **Pass:** violation rate `≤ δ_j + 3·sqrt(δ_j(1−δ_j)/R)` for every band, generator and `n_calib`. Also report tightness: certified coverage divided by oracle coverage at the same risk.

**T1-real (approximate; low power, sanity check).**
- Per wave-1 dataset with a frozen graph, inside the warden: pool `P = calib ∪ sealed`; `R = 1000` regime-respecting resplits into `C'` / `T'`; run the certifier on `C'`; on `T'` a violation is "Clopper–Pearson lower 95% bound on test risk `> α_j`."
- **Pass:** same slack criterion as T1-synth.

**T1-expensive.** Re-run the full search under 5 outer seeds (different outer splits), certify through the warden, check sealed. **Pass:** no band with a Clopper–Pearson lower bound above `α_j` in any seed.

**T1-forecast.** Long-run interval miscoverage within ±0.02 of target over at least 2000 steps; local-coverage plots; selective risk reported as empirical with block-bootstrap intervals.

**Invariants** (property tests): bands nest; `τ̂` non-decreasing in `α`; p-values monotone in `α`; `n_min` warnings fire; determinism; certify counter enforced.

### T2. It earns its complexity

Baselines per family:
- **B0**: best single-operator pipeline. Tabular and time series: AutoGluon preset (medium) or LightGBM; text: best of TF-IDF + LR and frozen-embedding + LR; image: frozen-encoder + LR; extraction: regex/CRF baseline; anomaly: supervised GBDT.
- **B1**: B0 + learned scorer + **the same certifier** (the fair baseline).
- **B2**: B0 + naive max-confidence threshold chosen on dev (uncertified). Report its violation rate on sealed to show why certification is needed.

Metric: certified coverage on sealed at each band (via warden), 5 outer seeds. AMX wins a (dataset, band) cell if mean `Δcov > max(0.02, SE)`.
**Pass:** wins on at least 60% of (dataset × tight band) cells and is never significantly worse (`Δ < −2·SE`) anywhere. Also run the backend A/B (section 9.8): if `ratchet` does not beat the best of `tpot` / `bandit` / `random` on a majority of datasets, record a decision-log entry on whether the agent adds value.

### T3. Reason codes are real

Detectors meet the targets in section 10.2 on the datasets that supply ground truth (CLINC150 out-of-scope, CIFAR-10H entropy, Covertype minority classes, injected label flips, injected invariant violations). Adult should come out dominated by `irreducible`.

### T4. Portable and cheap

- **Zero diffs** in `src/`, `claude-plugin/`, `programs/` between datasets. A CI job runs all wave-1 datasets from one commit using only `datasets/<name>/`.
- **Domain-blind lint:** forbidden-string scan (dataset names, column names, industry words from a maintained list) over those directories.
- **Budgets** respected and documented per dataset (initial: ≤ 4 h CPU-only wall clock; ≤ 5M agent tokens).
- **Seed stability:** std of certified coverage across 5 seeds `≤ 0.02` absolute or `≤ 10%` relative.
- **Reproducibility:** same seed + commit + lockfile gives the same graph hash and metrics (bitwise for CPU ops; stated tolerance for DL).

### T5. Domain-agnostic

Leave-one-**domain**-out. For each held-out domain (suite must span at least 5 domains), warm-start from the ledger of the other domains. **Pass:** experiments-to-90%-of-final-certified-coverage `≤ 70%` of cold start on at least 3 of 5 held-out domains, with certified coverage not worse than cold start beyond noise. Domain-blind lint clean.

---

## 13. Validation datasets

One dataset per task family, spread across domains. Loaders live in `datasets/<name>/` (`spec.yaml`, `loader.py`, `NOTES.md`). Data is **never committed**; loaders download to `~/.cache/amx/datasets`. Where relevant a loader exposes `truth()` for T3 ground truth. Every dataset gets a `LICENSE_NOTE.md`.

| Dataset | Family | Commit unit / ℓ | Domain | What it proves | Notes | Wave |
|---|---|---|---|---|---|---|
| Adult (UCI) | classification (tabular) | row / 0-1 | social, census | mixed types; substantial irreducible noise; known-negative check for LLM-suitability; inject label flips for `noisy_gold` | VERIFY license | 1 |
| California Housing | regression | row / error > tol | housing, geo | heteroscedastic noise; target piles up at a cap (the profiler must notice); interval-width abstention | via scikit-learn; VERIFY | 1 |
| BANKING77 | classification (text) | row / 0-1 | customer support | about 13,000 requests across 77 closely related intents, official test split of 3,080: confusable classes, candidate sets, per-class feasibility limit. Public LLM-side comparators exist (llm-scorekit, third-party Jev tests) for Phase B. | VERIFY license | 1 |
| CIFAR-10 + CIFAR-10H | classification (image) | row / 0-1 | vision | human label counts (about 50 judgments per image) on the 10,000 test images: ambiguity ground truth | CIFAR-10H is **CC BY-NC-SA 4.0**: research/internal evaluation only | 1 |
| GIFT-Eval **or** Monash subset | forecasting | series × horizon / error > tol | multi-domain | Subset with no single archive domain above one third. GIFT-Eval: 23 datasets, 7 domains, 10 frequencies, test windows strictly after training, a leaderboard type for agentic systems. Monash: open archive with statistical, ML and DL baselines. | GIFT-Eval repo is research-use only; choose per open decision O4 | 1 |
| CORD | extraction | field / 1 − match | receipts, finance | images with box-level OCR text and multi-level semantic labels; field-level commit; `independence_unit: group:doc_id` | **CC BY 4.0** | 1 |
| ADBench subset | anomaly | row / missed anomaly among committed-normal | multi-domain | 57 datasets, 30 algorithms; no unsupervised algorithm statistically dominates, which is the case for automatic selection | code BSD-2; check each dataset's source. Pin the subset (OQ); no single ADBench category above one third of it | 1 |
| CLINC150 | classification (text) | row / 0-1 | dialog | 150 in-scope intents plus 1,000 out-of-scope test queries: `novel` ground truth | VERIFY license | 2 |
| BeyondArena (TabArena) | classification / regression | row | multi-domain | IID, temporal and grouped tasks: shows where certificates fail under shift and exercises the regime audit | | 2 |
| Covertype | classification (tabular) | row / 0-1 | geo, ecology | 581k rows, 7 classes, one class under 1%: `sparse_support`, scale | VERIFY license | 2 |
| Ranking and detection sets | ranking / detection | | | R2 families (certifiers exist in the literature for ranked retrieval and for IoU/recall control in detection) | TBD | 3 |

Non-commercial or research-only data is for **evaluation of AutoMaX only**: no redistribution; clear with legal before any commercial use of derived artifacts.

---

## 14. Milestones

Sizes: S/M/L/XL are relative effort. **Do not skip a gate.**

### A0. Harness core, no agents (L)

- [ ] Repo scaffold: `uv`, ruff, mypy, pytest, hypothesis, pre-commit, CI, Makefile targets (section 15).
- [ ] `amx.spec`: pydantic models, validators, JSON-schema export to `docs/schemas/`.
- [ ] `amx.loss`: builtins `zero_one`, `err_gt_tol` (absolute or relative tolerance), `one_minus_f1`, `missed_anomaly`; custom loader; validators (7.9).
- [ ] `amx.data`: parquet/csv/jsonl loaders, `UnitFrame`, content-hash cache.
- [ ] `amx.split`: regimes `iid`, `grouped`, `temporal`; hash-locked manifests; vault client (`local_dir` mode only at this stage).
- [ ] `amx.cert`: binomial and Hoeffding–Bentkus p-values, `n_min`, fixed-sequence LTT, δ budgeting, nesting assertion, certify-call counter, guarantee objects, `bands.json` writer.
- [ ] ACI basic and rolling-origin evaluation for forecasting.
- [ ] `amx.sim`: four synthetic generators; T1 simulators (synthetic and real).
- [ ] `amx.report` skeleton; CLI: `doctor`, `profile` (modality, regime audit, feasibility), `split`, `simulate`.
- [ ] Tests: unit, property (nesting, p-value monotonicity), statistical (Monte Carlo validity of every p-value), e2e-small.

**Gate A0:** T1-synth passes on all generators, `n_calib` values and bands. T1-real-cheap passes on Adult (classification), California Housing (regression) and one series (forecasting) using trivial models. `make check` green.

### A1. Operators, scorer, baselines (L)

- [ ] Operator contract, registry, contract tests (8.3).
- [ ] Library v1:
  - rules and lookups: `ExactMatch`, `NearDupLookup`, `RuleSet` (mined from tree paths with a precision floor);
  - tabular: `LogisticHead`, `RidgeHead`, `GBDTHead`, `ForestHead`, `KNNHead`, optional AutoGluon composite;
  - text: `TfidfFeaturizer`, `FrozenTextEncoder`, linear and kNN heads;
  - image: `FrozenImageEncoder` + heads;
  - series: `SeasonalNaive`, `ETS/ARIMA`, `GBDT-on-lags`;
  - probabilistic: `IsotonicCalibrator`, `TemperatureScaler`, `VennAbers`, `SplitConformalSet`, `KNNDensityOOD`, `EnsembleVariance`;
  - combiners: `Mean/Vote`, `StackEnsemble`.
- [ ] Modality adapters with content-hash embedding cache (key: encoder id + version + input hash).
- [ ] Learned scorer + calibration (7.7). Single-stage selective predictors and a search over confidence scores.
- [ ] `amx eval` basic: K-fold OOF runner, metrics JSON (no search logic yet).
- [ ] Baseline runner B0 / B1 / B2.

**Gate A1:** baseline numbers for the wave-1 datasets (3 classification, 1 regression, 1 forecasting) recorded in the ledger. All library operators pass contract tests. T1-real passes on B1.

### A2. Graph, search, plugin, isolation (XL)

- [ ] Graph constructs and compiler (cascade, ensemble, router, loop, verifier); latency and size budgets.
- [ ] Fitness, acceptance rule (noise margin), ledger writes.
- [ ] Backends: `ratchet` (with supervisor), `evolve` adapter, `tpot` / `bandit` / `random` baselines.
- [ ] Warden and vault `container` mode; sandbox for `ops_user/`.
- [ ] Plugin v0: skills, agents, hooks, settings deny rules; `program.md` templates (classification, regression, forecasting).
- [ ] Red-team tests: the auditor attempts to read the vault, edit the protected core, call `amx certify`. All must fail.

**Gate A2:** T2 plus backend A/B on at least 3 datasets. T1-expensive on at least 3. Red-team passes. `claude plugin validate` clean.

### A3. Verifiers, reason codes, analyst, manifest, R1 completion (M to L)

- [ ] Verifiers and invariant discovery (10.3).
- [ ] Decomposed uncertainty via diverse ensembles; all seven reason detectors.
- [ ] Analyst procedure and deterministic taxonomy tagging; manifest v1 writer, `summary.json`, `interface.md`.
- [ ] Extraction (field-level) and anomaly end to end.

**Gate A3:** T3 passes. T1 for extraction and anomaly. **Manifest v1 frozen**, which unblocks Phase B.

### A4. Ledger, warm start, breadth, packaging (L)

- [ ] Fingerprinting, warm start, operator-family priors.
- [ ] Wave-2 datasets (CLINC150, BeyondArena, Covertype).
- [ ] Packaging (pip), docs, quickstart, examples; optional MCP server.
- [ ] R2 specification drafts for set-valued families.

**Gate A4:** T4 and T5 pass. Docs complete. Owner sign-off on license and distribution (O1).

**Definition of done (Phase A):** all gates pass; every guarantee in a report carries a truthful type and assumption list; reproducible from one command per dataset; README quickstart works on a clean machine.

---

## 15. Dev environment, tooling, test strategy

- **Env:** Python 3.12 (3.11 supported), `uv sync --extra dev`. Lockfile committed.
- **Quality:** ruff (lint + format), mypy strict on `amx.spec`, `amx.cert`, `amx.loss`; pre-commit; conventional commits; branch protection on the protected core (8.5).
- **Makefile targets:**
  ```
  make check            ruff + mypy + unit + property
  make contract         operator contract tests across the library
  make stat             synthetic-oracle T1 (seeded, marked slow)
  make e2e-small        tiny end-to-end run on a toy dataset
  make plugin-validate  claude plugin validate ./claude-plugin
  make redteam          isolation tests (agent must fail to read the vault)
  make portability      T4 diff + domain-blind lint across wave-1 datasets
  ```
- **Test pyramid:** unit; property (hypothesis); contract (operators); statistical (Monte Carlo, seeded, tolerance by binomial slack); e2e-small; red-team.
- **Synthetic oracle first.** Prove the certifier numerically on generators with known population risk **before** touching any real dataset.
- **Reproducibility:** seeds everywhere; record library versions and encoder ids in the run manifest; torch deterministic flags where feasible.
- **Performance:** cache embeddings; joblib parallelism; no GPU requirement in v1.
- **Logging:** `logging`, never `print`, in library code. Structured JSON logs for supervisor ingestion.

---

## 16. Risks and traps

| Risk | Why it bites | Mitigation |
|---|---|---|
| **ℓ is the attack surface** | A sloppy loss (tolerance too tight, exact match on fuzzy fields) makes certificates meaningless | validators (7.9); `--confirm-loss`; loss-distribution report |
| **Wrong split regime** | Time, groups, near-duplicates silently inflate coverage | exchangeability audit (8.2); regime downgrade with truthful guarantee type |
| **Structure search overfits dev** | More degrees of freedom than HPO | OOF + noise margin; Thresholdout-style switch; sealed folds; complexity penalty |
| **Composite and clustered units** | Field-level risk compounds at document level | `independence_unit`; group-level certification; report both |
| **Tight bands infeasible** | `n_min` exceeds committed calibration units | feasibility check at profile time; looser bands or larger calibration fraction |
| **Slice disparities** | Selective prediction can widen gaps between groups | per-slice reporting; `worst_slice_gap` in fitness; Mondrian only if data supports |
| **Forecasting guarantee is weaker** | ACI is marginal and long-run; selection by width breaks conditionality | label `holdout_empirical`; never upgrade |
| **Agent-written code leaks or hallucinates** | Target leakage, batch-statistics leakage, invented APIs | declarative graph + vetted library; contract tests; auditor; sandbox |
| **Guards bypassed** | Hooks are a policy layer; shell can bypass regexes; `@` references skip PreToolUse | container isolation + deny rules + hooks + red-team |
| **Long runs die** | Stop-hook continuation is capped; sessions end | external supervisor with bounded `claude -p` batches and ledger-based resume |
| **Cost blowups** | Tokens and compute | `--max-turns`, `--max-budget-usd`, budgets in TaskSpec, ledger accounting |
| **Encoder contamination** | Pretrained encoders may have seen benchmark text/images | record encoder id; include a non-pretrained baseline where feasible; caveat in report |
| **License exposure** | NC datasets | evaluation-only policy; `LICENSE_NOTE.md`; legal review before commercial use |
| **Cascade scorer mismatch** | Stage `i` scorer trained on units it never sees in deployment | recalibrate on units reaching the stage at `τ_0` (7.7) |
| **Scope creep** | R2 families, MCP, UI | decision D7, D16; gate discipline |
| **Non-reproducibility** | DL nondeterminism, library drift | locked env, recorded versions, stated tolerances |

---

## 17. Open decisions and decision log

### 17.1 Open decisions (owner answers at kickoff)

| # | Question | Default if unanswered |
|---|---|---|
| O1 | Distribution and license: public OSS or internal? Confirm the distribution name (`amx` may be taken on PyPI; fallback `automax`). | internal, name `amx` locally |
| O2 | Compute envelope: CPU only? GPU available? Is the AutoGluon dependency acceptable? | CPU only; AutoGluon as optional extra |
| O3 | Default demo bands and δ | `α = {0.5%, 1%, 2%, 5%}`, `δ = 0.10` |
| O4 | Forecasting dataset: GIFT-Eval (research-only) or Monash (open)? | Monash for development, GIFT-Eval for external comparison |
| O5 | Which agentic backend first in A2: `ratchet` or `evolve`? | `ratchet`, then `evolve` |
| O6 | Agent budget per run (USD, tokens) | `agent_max_usd: 150`, 5M tokens |
| O7 | Storage and PII policy for runs and ledger | public datasets only until a policy exists |
| O8 | Build an MCP server in A4? | no |
| O9 | Vault mode for real work: `container` or `remote`? | `container` |
| O10 | Frozen pretrained encoders: which families are acceptable license-wise? | sentence-transformers and DINOv2/CLIP-style, license-checked |

### 17.2 Decision log (locked unless changed here)

| ID | Decision |
|---|---|
| D1 | Name AutoMaX; package and CLI `amx`. |
| D2 | Phase A is LLM-free at inference; Claude Code is design-time only. |
| D3 | Unifying abstraction: commit unit + loss `ℓ ∈ [0,1]` + risk bands + one-parameter nested commit rule driven by a learned acceptability scorer. |
| D4 | Default certifier: fixed-sequence LTT with exact binomial / Hoeffding–Bentkus p-values. CRC for R2 set-valued families. ACI for time series. Guarantee types always labelled. |
| D5 | Frozen pretrained encoders allowed; identity recorded; contamination caveat reported. |
| D6 | Folds: `dev` (OOF), `calib` (budgeted certify calls), `sealed` (single touch). Labels and inputs of calib/sealed live in the vault. |
| D7 | R1 families: classification, regression, forecasting, extraction (field-level), anomaly. R2: multi-label, ranking, detection. |
| D8 | Search backends behind one interface; A/B against TPOT-style, bandit and random baselines is mandatory (T2). |
| D9 | Search space is declarative over a vetted operator library; new operator code only via the gated sandbox path. |
| D10 | Isolation is OS or container level (vault + warden). Hooks and deny rules are defense in depth only. |
| D11 | The LLM never grades. Acceptance is a deterministic rule on `amx eval` output. |
| D12 | Domain-agnostic: domain enters only through data, ℓ, constraints, priors. Enforced by lint. |
| D13 | Seven task-agnostic reason codes; manifest v1 is the Phase A to Phase B contract, frozen after A3. |
| D14 | Long runs are driven by an external supervisor with bounded `claude -p` batches, not by a single session. |
| D15 | Forecasting selective guarantee is labelled `holdout_empirical`; ACI provides long-run interval coverage only. |
| D16 | No MCP server in v1; CLI plus JSON files. |
| D17 | Domain-neutral vocabulary in contracts and docs (2026-10-05): anomaly family wording, loss builtin `miss_among_cleared` → `missed_anomaly`, taxonomy field `treatment_hypothesis` → `handling_hypothesis`, plus wording in 1.2, 6.6, 7.9, 9.8, 10.1, 10.4, 11.2, 13, 14, 16. Multi-domain archives are subset with a one-third cap per archive domain. Structural follow-ups (anomaly commit direction, ADBench subset) are open in `docs/OQ.md`. |

---

## 18. Kickoff

### 18.1 Checklist before the first session

- [ ] Python 3.12 and `uv` installed; `claude --version` recorded; `claude plugin validate` available.
- [ ] Repo created; `HANDOFF.md` and `CLAUDE.md` in the root; branch protection or CODEOWNERS on the protected core.
- [ ] Owner answers O1 to O4 (defaults acceptable for A0).
- [ ] For A0 the vault may be `local_dir` (no agents run yet). Decide O9 before A2.
- [ ] `ANTHROPIC_API_KEY` or subscription login confirmed for supervised batches (not needed for A0).

### 18.2 Paste-in kickoff prompt for Claude Code

```
You are the lead engineer on AutoMaX (amx). Read HANDOFF.md fully, then CLAUDE.md.

Goal for this session: milestone A0 only.

1. Before coding, reply with: (a) a file-by-file plan, (b) risks you see in
   sections 6 and 7, (c) any open decisions from section 17 you need answered now.
2. Then implement in this order, committing after each step with tests:
   amx.spec -> amx.loss -> amx.split -> amx.cert -> amx.sim -> CLI (doctor, profile, split, simulate).
3. Prove T1 on synthetic oracle data BEFORE touching any real dataset.
   Verify the Hoeffding-Bentkus formula against the LTT/RCPS papers and Monte Carlo.
4. Stop at the A0 gate. Report: test results, the T1 violation-rate table,
   guarantee types emitted, and open issues.

Do not build agents, the plugin, the search backends, or the warden container mode in this session.
Do not read or write anything under a vault path.
```

### 18.3 `CLAUDE.md` starter (keep short; hooks enforce what must always happen)

```markdown
# AutoMaX (amx): rules for Claude Code

- Read HANDOFF.md sections 1-8 before changing anything. Sections 6 and 7 are normative.
- Build in milestone order (HANDOFF section 14). Do not start the next milestone before its gate passes.
- NEVER read, list, grep or write anything under $AMX_VAULT, any path containing `vault/`,
  `folds/calib` or `folds/sealed`. Never print label columns of non-dev folds.
- During search sessions do not edit: src/amx/{cert,eval,split,warden,loss}/**,
  tests/{statistical,redteam}/**, claude-plugin/hooks/**, .claude/settings*.
  Changes there need a decision-log entry and owner approval.
- One change per experiment. Record the hypothesis and outcome with `amx ledger add`.
- LLM output never decides acceptance. Acceptance = `amx eval` JSON + the rule in program.md.
- Every statistical formula needs a Monte Carlo test. Label guarantee types truthfully.
- No domain strings (dataset names, column names, industries) in src/, claude-plugin/, programs/.
- Commands: `make check`, `make contract`, `make stat`, `make e2e-small`, `make redteam`.
- Style: ruff; mypy strict on spec/cert/loss; pydantic v2; logging, not print.
- Claude Code specifics change fast: verify hooks, skills, subagents and CLI flags against
  https://code.claude.com/docs before relying on them, and run `claude plugin validate`.
```

---

## 19. References

**Agentic AutoML and search**
- AutoML-Agent (multi-agent, retrieval-augmented planning, multi-stage verification): https://arxiv.org/abs/2410.02958
- MLE-STAR (web-search seeding, ablation-guided block refinement, ensembling): https://arxiv.org/abs/2506.15692
- MLZero / AutoGluon Assistant: https://github.com/autogluon/autogluon-assistant (arXiv:2505.13941)
- autoresearch (one mutable file, fixed budget, one metric, keep/discard): https://github.com/karpathy/autoresearch
- probabl-ai skills for agents (skore-based): https://github.com/probabl-ai/skills
- TPOT (DAG-structured genetic programming, multi-objective): https://github.com/EpistasisLab/tpot
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve · ShinkaEvolve: https://github.com/SakanaAI/ShinkaEvolve

**Selective prediction, conformal, risk control**
- Learn then Test: https://arxiv.org/abs/2110.01052
- Conformal Risk Control: https://arxiv.org/abs/2208.02814
- Selective Conformal Risk Control: https://arxiv.org/abs/2512.12844
- Selective classification can magnify disparities across groups: https://arxiv.org/abs/2010.14134
- Adaptive Conformal Inference under distribution shift: https://arxiv.org/abs/2106.00170
- Conformal time-series forecasting (review): https://arxiv.org/abs/2511.13608
- FrugalGPT (learned scorer + thresholds, LLM cascades): https://arxiv.org/abs/2305.05176
- llm-scorekit (automate-vs-defer decision layer): https://github.com/EnzoCanonero/llm-scorekit
- Autodistill (foundation-model labels to small supervised models): https://github.com/autodistill/autodistill
- Jev / System One decision model (Phase B comparator context): https://typesafe.ai/blog/introducing-system-one-models-and-jev

**Datasets**
- TabArena / BeyondArena: https://github.com/autogluon/tabarena (arXiv:2506.16791)
- GIFT-Eval: https://github.com/SalesforceAIResearch/gift-eval (arXiv:2410.10393) · Monash archive: https://forecastingdata.org/
- CORD: https://github.com/clovaai/cord
- ADBench: https://github.com/Minqi824/ADBench (arXiv:2206.09426)
- CLINC150: https://github.com/clinc/oos-eval (arXiv:1909.02027)
- CIFAR-10H: https://github.com/jcpeterson/cifar-10h
- BANKING77: Casanueva et al. 2020 (arXiv:2003.04807)

**Claude Code (verify at build time)**
- Plugins: https://code.claude.com/docs/en/plugins
- Hooks: https://code.claude.com/docs/en/hooks
- Skills: https://code.claude.com/docs/en/skills
- Subagents: https://code.claude.com/docs/en/sub-agents
- Headless / programmatic use: https://code.claude.com/docs/en/headless

---

## Appendix A. Pseudocode and formulas

### A.1 Cascade resolution

```python
def resolve(x, tau, stages):                   # stages in cascade order
    for st in stages:
        out = st.apply(x)
        g = st.scorer(out.signals)             # predicted loss, calibrated
        if g <= st.multiplier * tau:
            return Resolved(value=out.value, stage=st.id, score=g)
    return Abstain(reason=None)                # reason assigned later by amx.reasons

def commit_score(x, stages):
    return min(st.scorer(st.apply(x).signals) / st.multiplier for st in stages)
```

### A.2 Certification (fixed-sequence LTT)

```python
import math

def n_min(alpha, delta_j):
    return math.ceil(math.log(delta_j) / math.log(1 - alpha))

def certify(cal, graph, alphas, delta, grid, binary_loss, dev_cov):
    # dev_cov[g]: coverage of the frozen graph at grid[g], estimated on dev OOF (fixed before calibration)
    m, G, n_cal = len(alphas), len(grid), len(cal.X)
    dj = delta / m
    s = graph.commit_score(cal.X)                       # (n,)
    L = graph.loss_along_grid(cal.X, cal.y, grid)       # (n, G): loss of the answer given at each tau
    raw = []
    for a in alphas:
        need = n_min(a, dj)
        start = next((g for g in range(G) if dev_cov[g] * n_cal >= 1.25 * need), None)
        tau_hat, status = None, ("infeasible_on_dev" if start is None else "none")
        if start is not None:
            for g in range(start, G):                   # liberal-ward walk
                mask = s <= grid[g]
                n = int(mask.sum())
                if n < need:
                    status = "sample_size_limited"; break
                Rhat = float(L[mask, g].mean())
                if p_value(Rhat, n, a, binary=binary_loss) <= dj:
                    tau_hat, status = grid[g], "certified"
                else:
                    status = "risk_limited"; break
        raw.append((a, tau_hat, status))
    out, best = [], None                                # monotonize: tighter-band thresholds are valid for looser bands
    for a, t, st in raw:
        if t is not None:
            best = t if best is None else max(best, t)
        out.append((a, best, st))
    return out
```

### A.3 p-values (VERIFY)

```python
from scipy.stats import binom
import numpy as np

def p_binary(k, n, alpha):                    # H: R >= alpha
    return float(binom.cdf(k, n, alpha))

def h1(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

def p_hb(Rhat, n, alpha):                     # bounded loss in [0, 1]
    if Rhat >= alpha: return 1.0
    hoeff = np.exp(-n * h1(min(Rhat, alpha), alpha))
    bent  = np.e * binom.cdf(np.ceil(n * Rhat), n, alpha)
    return float(min(hoeff, bent))
```

### A.4 ACI update (forecasting intervals)

```python
alpha_t = alpha_target
for t, (y_t, interval_fn) in enumerate(stream):
    C = interval_fn(level=1 - alpha_t)
    err = 0 if (C.lo <= y_t <= C.hi) else 1
    alpha_t = alpha_t + gamma * (alpha_target - err)      # gamma default 0.005
```

### A.5 Fitness and acceptance

```python
def fitness(m):                                           # m = amx eval dev_oof metrics
    return (sum(w * b["coverage_lcb"] for w, b in zip(W, m["bands"]))
            - lam_lat   * np.log1p(m["cost"]["apply_ms_p50"])
            - lam_size  * np.log1p(m["cost"]["size_mb"])
            - lam_gap   * m["worst_slice_gap"]
            - lam_nodes * m["n_nodes"] / 10)

def accept(F_new, F_old, se_paired, eps=0.002, z=1.0):
    return (F_new - F_old) > max(eps, z * se_paired)
```

---

## Appendix B. Example TaskSpecs (abbreviated)

**Text classification**
```yaml
name: intents
data: {uri: data/intents.parquet, format: parquet, unit_id: id, inputs: [{name: text, kind: text}], target: {name: label, kind: categorical}}
task: {family: classification, commit_unit: row, loss: {kind: builtin, name: zero_one}}
bands: {alphas: [0.01, 0.02, 0.05], policies: [auto, auto, audit], delta: 0.10}
```

**Regression**
```yaml
name: value-regression
data: {uri: data/value.parquet, format: parquet, unit_id: id, target: {name: value, kind: numeric}}
task: {family: regression, commit_unit: row, loss: {kind: builtin, name: err_gt_tol, params: {tol: 0.25, relative: true}}}
bands: {alphas: [0.02, 0.05, 0.10], policies: [auto, audit, audit], delta: 0.10}
```

**Forecasting**
```yaml
name: series-forecast
data: {uri: data/series.parquet, format: parquet, unit_id: row_id, time_column: ts, group_columns: [series_id], target: {name: y, kind: series}}
task: {family: forecasting, commit_unit: series_horizon, loss: {kind: builtin, name: err_gt_tol, params: {tol: 0.10, relative: true, horizons: [1, 7, 28]}}}
splits: {regime: temporal}
bands: {alphas: [0.05, 0.10], policies: [auto, audit], delta: 0.10}
```

**Field extraction**
```yaml
name: doc-fields
data: {uri: data/fields.parquet, format: parquet, unit_id: field_id, group_columns: [doc_id], independence_unit: "group:doc_id", target: {name: value, kind: spans}}
task: {family: extraction, commit_unit: field, loss: {kind: builtin, name: one_minus_f1}}
bands: {alphas: [0.01, 0.03], policies: [auto, audit], delta: 0.10}
```

---

## Appendix C. Glossary

- **Unit / commit unit**: the smallest thing the system can answer or pass on.
- **Loss ℓ**: score in [0,1] of a committed answer against gold.
- **Band**: ceiling on selective risk; bands nest.
- **Tier**: `AUTO`, `AUDIT` or `RESIDUAL`, determined by the smallest band that resolves a unit.
- **τ**: maximum predicted loss at which the system commits; larger commits more.
- **Commit score**: `min_i ĝ_i(x) / m_i`; a unit commits iff it is at most τ.
- **Scorer ĝ**: learned, calibrated estimate of the loss a stage's answer will incur.
- **Selective risk / coverage**: mean loss over committed units / share of units committed.
- **Certificate**: statement that selective risk is at most α at level δ, with a guarantee type and assumptions.
- **Regime**: split and exchangeability structure (`iid`, `grouped`, `temporal`, `blocked`).
- **Vault / warden**: protected store of calibration and sealed data / the only process allowed to use it.
- **Operator / graph / stage / cascade**: typed component / typed DAG of operators / an answering step with a scorer / ordered stages that pass abstentions down.
- **Set A / Set B**: what the pipeline resolves / the residual handed to Phase B.
- **Reason code**: why a residual unit was not resolved.
- **Manifest**: the Phase A to Phase B handoff table.
- **Ratchet**: keep a change only if the deterministic evaluator says it is better by more than noise.

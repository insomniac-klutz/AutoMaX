# AutoMaX ROLLER: rolling build plan

ROLLER is the living plan for building AutoMaX: the milestone status board, the next action, a file-by-file plan for the current milestone, and the corrections and risks the build has adopted. It is updated after every step. `HANDOFF.md` stays the spec; `docs/OQ.md` holds the owner's decisions.

Last updated: 2026-10-07.

## Status board

| Milestone | State | Gate | Blocked on |
|---|---|---|---|
| Planning | **done** | review complete (4 lenses, adversarial verify, completeness critic) | none |
| A0 harness core, no agents | **ready to start** | T1-synth + T1-real-cheap + `make check` | G1 to G3 and Q1 to Q8 (defaults apply if unanswered) |
| A1 operators, scorer, baselines | not started | baselines in ledger, contract tests, T1-real on B1 | Q9 to Q13; encoder network access |
| A2 graph, search, plugin, isolation | not started | T2 + backend A/B, T1-expensive, red-team, plugin validate | Q14 to Q19; Docker host for the warden |
| A3 verifiers, reasons, manifest | not started | T3; manifest v1 frozen | Q20 to Q22; CLINC150 and Covertype loaders |
| A4 ledger, warm start, packaging | not started | T4, T5 | Q23 to Q24; G1 |

**Next action:** A0 step 1 (scaffold), as soon as G1 to G3 are answered or accepted at their defaults.

## Environment facts (cloud session, verified 2026-10-05)

- Python 3.11.15 is the default. `uv python install 3.12` works; uv 0.8.17; 4 CPUs; 15 GiB RAM.
- Claude Code 2.1.289 is installed. Use it for the 11.x VERIFY items in A2.
- The session runs as root with a Docker client but **no daemon**. `local_dir` vault gives no isolation here, and `container` mode cannot run here.
- **Reachable:** PyPI, GitHub raw/LFS/codeload, storage.googleapis.com, S3, conda.
- **Blocked:** UCI, OpenML, figshare, Hugging Face, zenodo, forecastingdata.org, arXiv, download.pytorch.org, Kaggle, Google Drive.
- Verified gate-data mirrors:
  - Adult: AutoGluon S3, 48,842 rows (52 exact duplicate rows).
  - California Housing: Keras GCS `.npz`, 20,640 × 8, 965 rows at the 500,001 cap.
  - Series: ElectricityLoadDiagrams via the LSTNet GitHub mirror, 26,304 hourly points × 321 clients.
- Also reachable for later milestones: BANKING77 (CC BY 4.0), CLINC150 (CC BY 3.0), CIFAR-10H counts, ADBench `.npz` files.
- The §7.3 HB p-value matches the LTT authors' reference code (`aangelopoulos/ltt` `core/bounds.py`). The paper text itself could not be checked because arXiv is blocked.
- PyPI `amx` is taken (an unrelated project); `automax` is free.

## A0 corrections adopted (builder-owned, no owner decision needed)

These are errors or gaps in `HANDOFF.md` that the review confirmed. Each will be logged as a decision (D18 onward) in A0 step 1 and implemented as described. Implementations deviate from the handoff's pseudocode wherever these items say so.

| ID | Handoff | Correction |
|---|---|---|
| C1 | A.2, A.3 | The binomial p-value takes the **integer** loss count `k = sum(L)`, never `n·mean`. The float can come out as k−ε; scipy then floors it, which is anti-conservative. Example: n = 374, k = 1, α = 1% gives p = 0.023 instead of the correct 0.111. HB uses exact integer sums. A regression test is added. |
| C2 | 7.2 step 6, 12 T1 | After cross-band monotonisation, band j is only bounded by Σ_{k≤j} δ_k. T1 gates on (i) raw per-band rates at δ_j and (ii) the family-wise rate at δ. The certificate keeps both the raw τ̂_j and the monotonised τ̂'_j. |
| C3 | A.2 | A band that inherits a tighter band's τ gets status `inherited` instead of keeping a stale `risk_limited` status. |
| C4 | A.1 vs 5.2 | `score` is the commit score s(x) = min_i ĝ_i/m_i, not the answering stage's ĝ. |
| C5 | 7.3 | p-value Monte Carlo matrix: Bernoulli losses, two-point {0, 0.5} and {0, 0.99}, and Beta losses; n ∈ {n_min, 400, 2000}; δ ∈ {0.1, 0.025, 0.0083}. Assert rate ≤ δ + 3·MC SE, plus a tightness floor and an end-to-end LTT test. |
| C6 | 12 T1-synth | Population risk is computed by Rao–Blackwell averaging of the known conditional risk r(x), not from sampled labels. A cell where \|R(τ) − α\| < 4 SE at a selected τ is marked indeterminate. |
| C7 | 6.5, 7.10, 12 | Clopper–Pearson bounds only for binary ℓ; HB/Bentkus bounds for fractional ℓ. Slices with n < n_min(2α, 0.05) are labelled "insufficient n". |
| C8 | 6.1 | Spec validation additions:<br>• fraction keys must be exactly {dev, calib, sealed}, with \|Σ − 1\| ≤ 1e-9;<br>• policies must be `auto*` followed by `audit*`, with at least one `auto`;<br>• δ ∈ (0, 0.5];<br>• `amx_version` is renamed `spec_version`;<br>• `group:<col>` must name a column in `group_columns`;<br>• relative `err_gt_tol` gets an absolute floor near y = 0;<br>• the cross-reference "see 7.4" becomes "see 7.5". |
| C9 | 5.3, 7.4, 8.2 | `profile` runs in two passes: a label-free pass on all data (time, groups, duplicates, modality) before `split`, then a feasibility pass on dev only after it. Label statistics come from dev only. The n_min check moves out of pydantic validation. |
| C10 | 4.3 vs 8.5 | One protected-core list, the 8.5 list including `loss`. Used by CODEOWNERS, the guard and CLAUDE.md. |
| C11 | 4.1 | Add `pyyaml` and `matplotlib` to core dependencies and `scipy-stubs`, `pandas-stubs`, `types-PyYAML` to dev. Content hashes are computed from Arrow buffers, not pandas dtypes. Pin one pandas major. |
| C12 | 7.5, A.4 | ACI: α_t ≤ 0 gives an infinite interval and α_t ≥ 1 an empty one; both are counted and reported. Updates are per horizon with delayed feedback err_{t−h}. A property test checks the deterministic bound. |
| C13 | 5.1, 8.2 | Dev data lives at `runs/<id>/data/dev.parquet`. The split manifest lists dev ids, plus only counts and HMAC digests (vault-held key) for calib and sealed. |
| C14 | 7.4, 8.2 | Generic profiler checks: a label-noise floor (bands below it are infeasible) and a constant-predictor check (bands a majority predictor already meets are flagged). |
| C15 | 7.2 | A `cert:` block in the TaskSpec (grid, start factor, `cert_mode`, τ_0) is hashed at split. `certify` refuses CLI overrides. The Bonferroni mode is diagnostic only and never supplies the released τ̂. |
| C16 | 8.1 | OOF folds follow the outer regime: GroupKFold over groups and duplicate clusters, or forward-chaining with an embargo for temporal data. A test checks that fold assignment refines the cluster ids. |
| C17 | 6.5 | Coverage at τ̂ is an estimate. Its CI comes from a DKW band that is uniform in τ, or it is labelled descriptive. The field is renamed `coverage_at_certified_tau`. |

## A0 plan (file by file)

Order follows kickoff 18.2, with `amx.data` inserted because `split` needs it. Commit after each step with its tests. The S/M/L sizes are relative.

| # | Module | Files | Deliverable | Key tests | Deps | Size |
|---|---|---|---|---|---|---|
| 1 | scaffold | `pyproject.toml`, `uv.lock`, `.python-version` (3.12), `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.github/CODEOWNERS`, `.gitignore` (+`runs/`, data caches), `CLAUDE.md`, `src/amx/{__init__,py.typed,_log}.py`, `tests/conftest.py` | Distribution `automax` (G3); import and CLI `amx`; requires-python ≥ 3.11; ruff (T20 bans `print` in `src/`); mypy strict on spec/cert/loss; logging only. Decision entries D18 onward for C1 to C17 | `uv sync --locked --extra dev` on 3.11 and 3.12; `make check` green on the empty package; `amx --help` exits 0 | none | M |
| 2 | `amx.spec` | `spec/{enums,models,validators,loader,hashing,schema}.py`, `docs/schemas/taskspec.schema.json` | Frozen pydantic v2 models (`extra='forbid'`) for every 6.1 block plus the `cert:` block (C15); validators per C8; canonical hash | a passing and a failing case for each rule; the four Appendix B specs load; YAML round-trip property; schema drift test | 1 | M |
| 3 | `amx.loss` | `loss/{base,builtins,registry,custom,validate,squash}.py` | Vectorized `Loss` returning float64 in [0, 1], with `is_binary` derived from params. Builtins: `zero_one`, `err_gt_tol` (abs/rel, floor, horizons), `one_minus_f1` (+ exact), `missed_anomaly`, `anomaly_cost` (Q8). Custom loader gated by `--confirm-loss`. 7.9 validators and the loss-distribution report | property: range [0, 1] and ℓ(y, y) = 0; validator rejects ℓ = 1.5 and ℓ(y, y) = 0.2; relative tol at y = 0 is finite | 2 | M |
| 4 | `amx.data` | `data/{unitframe,io,hashing,cache,fetch}.py` | `UnitFrame` (ids, Arrow inputs, optional target, groups, time; `take`, `without_target`, order-independent `content_hash`); parquet/csv/jsonl loaders plus the `datasets/<name>/loader.py` protocol; `fetch(urls, sha256)` with mirror fallback | duplicate `unit_id` rejected; hash invariant to row shuffle; `without_target` has no target buffer; fetch falls back on a hash mismatch (local server fixture) | 2 | M |
| 5 | `amx.split` | `split/{regimes,dedupe,oof,manifest,vault,counter}.py` | Regimes `iid`, `grouped` and `temporal` (embargo ≥ max horizon + lag); `blocked` deferred. Exact-duplicate clusters go to one fold and calib/sealed matches are dropped (Q7). OOF folds per C16. Manifest per C13. `local_dir` vault (0700/0600). Certify counter that is safe across processes | property: no group straddles folds; temporal ordering with embargo; same seed gives the same manifest hash; manifest holds no target values; counter refuses max + 1 | 2, 4 | L |
| 6 | `amx.cert` | `cert/{pvalues,nmin,grid,ltt,budget,bounds,slices,guarantee,certificate,writer}.py`, `docs/schemas/bands.schema.json` | Pure functions, no I/O. p-values per C1 and C5; `n_min`; fixed log grid (G = 200) hashed into the certificate; start index; vectorized fixed-sequence walk (cumsum gives n_g, k_g) with statuses certified, risk_limited, sample_size_limited, infeasible_on_dev and inherited; Bonferroni-diagnostic mode; group-level variant (Q2); δ budget per Q1; guarantee objects; `bands.json` writer | reproduces every 7.4 cell and the 895 figure; equals a corrected A.2 reference on 1k random cases; HB equals the reference formula; properties: nesting, τ̂ non-decreasing in α, p monotone; MC matrix (C5) | 1, 3 | L |
| 7 | `amx.cert.aci` | `cert/{aci,rolling,block_bootstrap}.py` | Split-conformal per horizon, then ACI per C12; rolling-origin evaluation with embargo; block-bootstrap CI; forecasting selective risk is always `holdout_empirical` (Q6) | deterministic bound on adversarial 0/1 sequences; MC on AR(1)+season (miscoverage and infinite-interval share reported); label never upgraded | 6 | M |
| 8 | `amx.sim` | `sim/{generators,oracle,t1_synth,tables}.py` | Generators: (a) Gaussian mixture, (b) heteroscedastic regression, (c) AR(1)+seasonal with shift, routed to the T1-forecast path, (d) clustered units at group level. Each has a closed-form scorer and Rao–Blackwell population risk (C6). The T1-synth runner outputs raw, monotonised and family-wise rates, tightness, and vacuous/indeterminate cells. Seeds are pre-registered in the decision log | oracle risk matches 1e6-sample MC within 4 SE; fast smoke (R = 100) in `make check`; full grid in `make stat` | 6, 7 | L |
| 9 | trivial-model shim | `baseline/{trivial,artifact,resolve}.py` (moved from `sim/`, `warden/`; D22) | `ScoredPredictor` (fit on dev; `predict`; `commit_score`; `dev_cov` from 5-fold dev OOF). Classification: logistic regression with s from isotonic of 1 − max-prob. Regression: quantile-pair width mapped by isotonic. Series: seasonal-naive + split-conformal. Frozen with joblib plus a sha256. Provisional, replaced in A1 | fit only ever receives dev ids; same seed gives the same artifact hash; `dev_cov` non-decreasing in τ | 3, 4, 5 | M |
| 10 | `amx.profile` | `profile/{modality,audit,feasibility,recommend,fingerprint}.py` | Two passes per C9; exchangeability audit (time, groups, duplicates, cap pile-up, imbalance); feasibility with `--force` / `--allow-small`; C14 checks; certifier and guarantee selection per 7.1; `profile.json` | timestamp leads to `temporal`; repeated ids lead to `grouped`; cap flagged; infeasible band needs `--force`; n < 3000 needs `--allow-small` | 4, 5, 6, 9 | M |
| 11 | `amx.report` | `report/{bands_md,frontier}.py` | `bands.md` rendered from JSON only; guarantee type and assumptions; the 7.6 never-claimed list; warnings; slices; `frontier.png` (Agg backend) | golden file; "certified" absent for `none` / `holdout_empirical`; every number appears in the input JSON | 6 | S |
| 12 | `amx.warden` (minimal, local) | `warden/{token,runner,certify,t1_real}.py` | Freeze token never in the agent env. Runs the artifact in a subprocess that receives inputs only, computes losses itself, certifies and increments the counter. `simulate --real` implements T1-real-cheap per Q3 (calib-only resplits, aggregates only). Sealed touch log | refuses without a token; counter enforced across processes; subprocess payload has no target column; output has no per-unit gold | 5, 6, 9 | M |
| 13 | CLI | `cli.py`, `doctor.py` | `doctor [--json]` (versions, extras, vault mode and permissions, root warning, token absent from env), `profile`, `split`, `simulate --t1 --synthetic \| --real`. JSON logs and defined exit codes | CliRunner smoke on a toy spec; doctor JSON schema; `--real` without a token exits non-zero | 8, 10, 11, 12 | M |
| 14 | datasets + e2e-small | `datasets/{adult,california_housing,electricity_client}/{spec.yaml,loader.py,NOTES.md,LICENSE_NOTE.md}`, `tests/e2e/`, `tests/unit/test_domain_blind.py` | Mirror lists with pinned sha256 (Q4); loss and bands per Q5 (owner-confirmed); offline toy end-to-end run; domain-blind lint (dataset and column names absent from `src/`) | network-marked loader shape and hash checks; `make e2e-small` offline; domain-blind lint | 4, 13 | M |
| 15 | Gate A0 | `docs/gates/A0.md`, decision log | `make check`, `make stat` (T1-synth table), T1-real-cheap on the three datasets; guarantee types emitted; open issues | the gate itself | 8, 12, 13, 14 | S |

### Gate A0 (as amended by C2, C6, Q3, Q6, D21, D24)

1. `make check` is green.
2. T1-synth over generators (a), (b), (d) × n_calib {500, 2000, 10000} × bands:
   - raw per-band violation rate ≤ δ_j + 3·sqrt(δ_j(1 − δ_j)/N_rep);
   - family-wise rate ≤ δ + the same slack;
   - vacuous and indeterminate cells reported, not counted as passes (cell verdict `gate_pass`);
   - per generator, some band certified in at least 10% of reps at n_calib = 10000.
3. T1-forecast on the stationary variant of generator (c): |miscoverage − α| ≤ 0.02 (empirical target) and infinite + empty interval share ≤ 0.01 per horizon. The shifted path is reported with its one-sided ACI bounds, not gated (D21).
4. T1-real-cheap (calib-only resplits, after the certify call) on Adult, California Housing and the electricity client: no gross failure, and some band released in at least 10% of resplits per dataset.

Results: `docs/gates/A0.md`.

## Later milestones (outline; expanded when the previous gate passes)

**Carried forward from A0 (found while building or reviewing A0).**
- Start-rule power: the 1.25·n_min start certifies tight bands rarely (generator (a): α ≤ 1% never, α = 5% in 57% of reps at n_calib = 10000). Evaluate the dev-simulated start rule (review S-start-rule-low-power) in A1; validity is unaffected (OQ Q26).
- `dev_cov` comes from single-fold OOF scores while the artifact is the bag of fold models (Q13). The deployed score distribution differs, so start points can miss (California's 5% and 10% bands stopped `sample_size_limited`). A1's cross-fitted scorer should estimate `dev_cov` for the deployed bag.
- Anomaly family: `missed_anomaly` only bounds the intended quantity if the commit rule commits units predicted normal only; enforce that in the A3 graph and validate `normal_label` against dev gold.
- Release mode (A2): bind tokens to one command and consume them on use, read them from a prompt or file rather than argv, refuse re-splitting a dataset that already has a release, and run the warden in container mode.

**A1: operators, scorer, baselines.**
- Operator protocol with `apply(x, ctx)` and declared `signal_names`.
- Contract tests. Batch invariance compares discrete outputs exactly and floats within a per-op tolerance. Commit decisions must be batch-invariant.
- Library v1; embedding cache.
- Learned scorer:
  - nested cross-fitting, so `ctx.oof()` is inner-OOF and eval metrics come from cross-fitted scores;
  - a strictly monotone s(x), with ties broken by the raw signal;
  - no-signal handling: a constant scorer at the OOF mean, flagged `irreducible` only if that mean exceeds α_m.
- `amx eval`; B0/B1/B2 plus `B_const`.
- Artifact fitted per Q13.
- **Owner first:** Q9 to Q13; encoder network access.
- **Gate concerns:** without encoders, text B0 is TF-IDF only and CIFAR-10 waits; T1-real on B1 must not read sealed (Q18).

**A2: graph, search, plugin, isolation.**
- Graph compiler: stage id = cascade entry, a per-stage `scorer:` block, verifier failure sets ĝ = +∞, loops only inside `fit`.
- Fitness:
  - the coverage term is expected certified coverage, from simulating the warden procedure on calib-sized bootstraps of dev OOF;
  - every evaluation counts toward N_max;
  - a hidden `search_reserve` fold.
- `amx eval --against <incumbent>` emits {ΔF, SE, accepted}. The harness, not the agent, does git commits and ledger writes.
- Backends.
- Warden in container mode:
  - `split` runs in the warden and the agent never mounts `data.uri` or loader caches;
  - signed certificate; hash-pinned artifact that `amx.load` verifies;
  - the TaskSpec and custom loss are snapshotted into the vault at split.
- Sandbox: the fit sandbox sees only the train fold; lint caps literal size in `ops_user/`.
- Search-time controls:
  - a Bash allowlist and WebFetch/WebSearch denied;
  - seed isolation, with release budgets keyed by dataset content hash;
  - `claude -p` launched with `env -i`.
- Plugin v0.
- `docs/claude-code-facts.md`: every 8.7, 9.5 and 11 claim, backed by red-team tests against the installed `claude --version`.
- **Owner first:** Q14 to Q19; a Docker host.
- **Gate concerns:** cost (Q15) and multiplicity in T2 (Q19).

**A3: verifiers, reasons, analyst, manifest.**
- Inference-time reason codes are label-free and frozen in the artifact; gold-dependent codes (`noisy_gold`) are audit-time, manifest only.
- The analyst sample comes from dev-OOF residuals. The warden applies the analyst's tagging rules to calib/sealed and writes the manifest after the calib budget closes, with a per-fold column allowlist and no sealed gold.
- Invariants carry provenance.
- Manifest v1: per-band value and stage columns, `handling_hypothesis`, `gold_exposed_to_phaseA`, and a Phase B dev/test split.
- Extraction units per Q21; anomaly per Q8; ADBench per Q20.
- The CLINC150 and Covertype loaders move here from A4 (Q22).
- Label-flip injection runs on a separate copy, dev only.

**A4: ledger, warm start, breadth, packaging.**
- Per-run experiment store plus a global ledger written only after finalisation.
- Fingerprints without domain features; the domain used only as T5 metadata (Q23).
- Wave-2 datasets; packaging as `automax`; docs and quickstart.
- T4 and T5 with the amended criteria (Q19, Q23).

## Risk register (additions from the review; the handoff's section 16 still applies)

| Risk | Where | Mitigation (adopted or proposed) |
|---|---|---|
| Adaptive certify calls invalidate the union bound | 7.11 | Q1 default: one call per calib fold; no calib feedback before the budget closes |
| Calib and sealed labels reach agents through the manifest, analyst, raw `data.uri` or loader cache | 6.6, 10.4, 8.4 | warden-side manifest, dev-only analyst sample, `split` inside the warden (A2) |
| Public-benchmark labels reachable over the network or from LLM memory | 13, 8.6 | egress allowlist for agent containers; lint on literal size; planted lookup-table red-team test; auditor coverage-jump check against B1 |
| Search optimises dev coverage the warden can never certify (n_min cliff) | 7.8, 9.3 | fitness uses simulated certified coverage |
| Outer seeds leak across runs | 12 T2/T4 | per-seed run tree, git repo, session and ledger isolation |
| Label noise or constant predictors make bands infeasible or trivial | 7.4 | C14 profiler checks; `B_const` baseline |
| Gate data hosts blocked | 13 | pinned mirrors (Q4) |
| No isolation in cloud sessions | 8.4 | cloud sessions for code only; vault work on an owner-controlled Docker host (Q16) |

## Update log

| Date | Change |
|---|---|
| 2026-10-05 | Created from the four-lens review. Planning done; A0 ready. `HANDOFF.md` 0.1.1 adds domain-neutral wording (D17). |
| 2026-10-07 | A0 built on OQ defaults (D18) and corrections C1-C17 (D19): steps 1-15. Data/split, ACI/simulators and report/baseline were built in parallel worktrees, each adversarially reviewed with its findings fixed; an integration review of the orchestrator-written modules found a slice leak of calibration gold values, a grouped auto path certified at unit level, unimplemented call policies and budget bypasses, all fixed with tests (commit e986e1e). Decisions D20-D25. Gate results in `docs/gates/A0.md`. |

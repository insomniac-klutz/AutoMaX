# AutoMaX open questions (OQ)

Owner decisions needed to build AutoMaX as specified in `HANDOFF.md`. Status: **open**, 2026-10-05.

**How to answer:** reply with the ID and a choice (for example `Q1: A`, `Q5: default`). A question you don't answer takes its **default** when its milestone starts. The default is then logged in `HANDOFF.md` section 17.2 as a provisional decision.

**Where these came from:** a four-lens review of the handoff (statistics, contracts/leakage, domain-neutrality, build feasibility). Every finding was checked by an adversarial verifier, then a completeness critic looked for gaps. Findings the builder can fix without you are in `docs/ROLLER.md` ("A0 corrections") and are not repeated here.

| Tier | When it must be answered | IDs |
|---|---|---|
| 0 | now (governance) | G1 to G3 |
| 1 | before A0 code | Q1 to Q8 |
| 2 | before A1 | Q9 to Q13 |
| 3 | before A2 | Q14 to Q19 |
| 4 | before A3 / A4 | Q20 to Q25 |
| new | raised by the A0 build (2026-10-07) | Q26 to Q28 |

---

## Tier 0: governance (now)

**G1. The repo is public, but O1 defaults to "internal".**
`insomniac-klutz/AutoMax` is public and already has an Apache-2.0 LICENSE. `HANDOFF.md` is pushed on branch `claude/automax-base-v1-vo0mv6`, so it is publicly readable now.
- Options: (a) public OSS under Apache-2.0; (b) make the repo private and keep AutoMaX internal.
- **Default: (a).** This matches the existing LICENSE. If you choose (b), switch visibility before more is pushed.

**G2. `maestro`, the default branch, is unprotected, but the design relies on protection.**
The protected core (`src/amx/{cert,eval,split,warden,loss}`, `tests/{statistical,redteam}`, `claude-plugin/hooks`, `.claude/settings*`) depends on branch protection and CODEOWNERS (HANDOFF 8.5, 18.1).
- **Default:** I add `.github/CODEOWNERS` in A0 step 1. You enable "require code-owner review" on `maestro`. PRs target `maestro`.

**G3. The PyPI name `amx` is taken by an unrelated project. `automax` is free.**
- **Default:** distribution `automax`; import package and CLI stay `amx` (the O1 fallback).

---

## Tier 1: before A0 code

**Q1. Certify-call regime and the δ budget.** *Blocks `amx.cert` and every n_min number.*
The doc gives two incompatible rules:
- 2.1, the 6.5 example and the 7.4 table use δ_j = δ/m.
- 7.11 also divides by `max_certify_calls`, which would require 477 committed units instead of 368 at α = 1%.

Worse, 7.11 lets later calls run on the *same* calib fold after results are visible. A graph changed after seeing calib output is a function of calib, so the union bound no longer holds. A Monte Carlo check showed a 0.0118 violation rate against a nominal 0.0083.

- (A) **One certify call per calib fold**, δ_j = δ/m. Any change after the call needs fresh calib, the same rule as `--new-sealed`.
- (B) Pre-register K graphs and certify them in one call at δ/(mK); release the best certified one.
- (C) Adaptive calls on K disjoint calib slices, at δ/m per call, with the released call fixed in advance.
- **Default: (A).** Keeps 2.1, 6.5 and 7.4 as written; (C) is opt-in for owners who want retries. In every option, no calib-derived file (certificate, frontier, calib manifest rows, calib gold) reaches agent-visible paths until the calib budget is closed.

**Q2. Group-level certification estimand.**
With `independence_unit: group:<col>`, the certified quantity is the **group-weighted** selective risk, not the unit-level R(τ) in 2.1. In addition:
- ℓ_g is fractional, so the binomial test is invalid and Hoeffding–Bentkus (HB) is mandatory.
- n_min, start_j and coverage must count groups.
- Regime `grouped` with `independence_unit: unit` violates i.i.d.
- **Default:** keep group-weighted, force HB, count groups, add `estimand` to the certificate, and make `regime: grouped` imply group-level certification. Unit-weighted risk via a ratio test stays an option.

**Q3. "T1-real-cheap" (Gate A0) is undefined, and the kickoff bans vault access.**
- **Default:**
  - Definition: T1-real with 200 resplits of **calib only**. It never reads sealed. It is descriptive and gates only on gross failure.
  - Models: the A0 trivial models (see ROLLER step 9).
  - Data: run on a *benchmark* split seed that is never reused for a release.
  - Who runs it: the builder runs `amx simulate --real` and sees aggregates only.
- Answer "owner runs it" if you want to hold the freeze token yourself.

**Q4. Gate data sources: the canonical hosts are blocked here.**
The egress proxy rejects UCI, OpenML, figshare (sklearn's California Housing), Hugging Face, zenodo and forecastingdata.org. GitHub raw, Google Cloud Storage, S3 and PyPI work.
- (a) Use mirrors pinned by sha256, cross-checked against each other:
  - Adult: AutoGluon's S3 copy, 48,842 rows (UCI, CC BY 4.0).
  - California Housing: the Keras GCS `.npz`, 20,640 rows, same StatLib origin.
  - Series: one client of UCI ElectricityLoadDiagrams via the LSTNet GitHub mirror, 26,304 hourly points, CC BY 4.0 at source.
- (b) Allowlist the canonical hosts in the environment network policy.
- **Default: (a)**, with mirror provenance recorded in each `datasets/<name>/LICENSE_NOTE.md`.

**Q5. Loss parameters and bands for the gate datasets.**
These go in the ℓ slot, so per 7.9 they are yours to confirm (`--confirm-loss`). With trivial models, the default bands (0.5% to 5%) leave several gate cells vacuous. The gate will also require at least one certified band per dataset.
- **Default (provisional):**
  - Adult: `zero_one`, bands {1, 2, 5, 10}%.
  - California Housing: `err_gt_tol`, relative tol 0.25, bands {5, 10, 20}%. Capped targets (965 rows at 500,001) are kept as-is, reported as a slice, and the assumption "risk measured against top-coded gold" is stated.
  - Series: `err_gt_tol`, relative tol 0.10 with an absolute floor near zero, horizons {1, 24}, bands {5, 10}%.

**Q6. Forecasting: one commit mechanism, and the role of ACI.**
- 7.1 commits via LTT on a calibration window; 7.5 commits when the half-width is at most τ_tol.
- ACI needs online label feedback, which contradicts "no online learning in production" (1.4) and the deterministic artifact (5.2).
- T1-forecast's ±0.02 at 2,000 steps is not implied by the ACI bound at γ = 0.005; that needs about 9,050 steps.
- The regime-shift generator (c) cannot pass a δ_j gate.
- **Default:**
  - Single mechanism: scorer, then τ grid, then LTT on the window, labelled `holdout_empirical`.
  - ACI is used for interval evaluation only, not shipped as online state.
  - T1-forecast's ±0.02 is labelled an empirical target, and the share of infinite or empty intervals is reported.
  - Generator (c) is gated on T1-forecast only.

**Q7. Near-duplicates and the certified population.**
Clustering near-duplicates into one fold and dropping calib matches restricts the certificate to "units with no near-duplicate in dev". The example `NearDupLookup` stage then never commits on calib, so its deployment commits are uncertified.
- (a) Keep the rule. State the restriction in `guarantee.assumptions` and mark lookup-stage commits uncertified in the report.
- (b) Where duplicates genuinely recur in deployment (iid), split by unit and keep them.
- **Default: (a)** globally, with a per-dataset override in `spec.yaml`.

**Q8. Anomaly family: commit direction (domain bias).**
The anomaly family only commits "normal" and only penalises misses. That is a rule-out (screening) design. Generic alerting also needs to commit "anomaly" with a false-alarm cost. Separately, in any family, a band whose α is at or above the base rate is trivially certifiable by a constant predictor.
- **Default:**
  - Ship two builtins: `missed_anomaly` (one-sided, as in the doc) and `anomaly_cost` (two-sided, cost-weighted 0/1, normalised to [0, 1]). An anomaly TaskSpec must name one; there is no implicit default.
  - A required `normal_label` parameter.
  - A family-agnostic profiler check that flags bands a constant predictor already meets, plus a `B_const` baseline in T2.

---

## Tier 2: before A1

**Q9. Compute, encoders and network (O2, O10).**
Hugging Face, `download.pytorch.org` and `dl.fbaipublicfiles.com` are blocked. Frozen text and image encoders cannot be fetched, and PyPI's torch is the large CUDA build.
- **Default:** CPU only, AutoGluon as an optional extra. You allowlist `huggingface.co` and `download.pytorch.org`, or supply the weights. Without that, the text B0 is TF-IDF + LR only and the CIFAR-10 image track waits.

**Q10. Forecasting dataset (O4).**
Both Monash (zenodo) and GIFT-Eval (Hugging Face) are blocked here.
- **Default:** develop on the ElectricityLoadDiagrams client plus an M4 subset from GitHub mirrors. Add the GIFT-Eval comparison when it is allowlisted.

**Q11. Use official test splits as sealed.**
- **Default:** yes for BANKING77 and CIFAR-10. This keeps the Phase B comparators valid and keeps CIFAR-10H ambiguity truth in one fold. Manifest v1 also gets a Phase B dev/test split of residual units and a `gold_exposed_to_phaseA` column.

**Q12. Fairness slices on Adult.**
- **Default:** declare `sex` and `race` as `watch_slices` in `datasets/adult/spec.yaml` (the data slot, so the domain-blind lint stays clean), report per-group selective risk, and do not list them in `forbidden_inputs` unless you say so.

**Q13. How the frozen artifact is fitted.**
- (a) A bag of the K dev fold models, so scorer signals match deployment.
- (b) Refit on all of dev, which shifts the signal distribution the scorer was trained on.
- **Default: (a).**

---

## Tier 3: before A2

**Q14. First agentic backend (O5).** **Default:** `ratchet`, then `evolve`.

**Q15. Program-level budget (extends O6).**
O6 caps each run, but the gates imply about 70 full agentic searches: T2 with backend A/B, T1-expensive, T4 seed stability and T5.
- **Default:** T2 outer seeds, T1-expensive seeds and T4 seeds are the same runs, and T5's cold arm reuses T4 runs. Phase A is capped at 70 agentic runs × $150 ≈ $10.5k, at 150 experiments per run (the 6.1 example's 300 is the per-run hard cap).

**Q16. Vault mode and where A2 runs (O9).**
`container` mode needs a Docker daemon. This cloud sandbox has none and runs as root, so `local_dir` gives no isolation here.
- **Default:** `container` mode on a machine you control (or a CI runner). Cloud sessions are used for code only, never for search against a vault.

**Q17. Dev data goes to the model provider (extends O7).**
By design, dev inputs and labels reach the LLM through agent tool calls.
- **Default:** public datasets only. Add a TaskSpec `agent_data_access: full | schema_and_aggregates` mode before any private data is used.

**Q18. Release mode vs benchmark mode.**
Sealed is single-touch per release, yet T2, T1-expensive and the A/B runs read sealed repeatedly.
- **Default:** two modes.
  - Release: a human-held freeze token, single-touch sealed.
  - Benchmark: outer-seed resplits, each with its own calib and sealed folds. No release certificate is issued. A harness holds the token and the builder sees pass/fail tables only.

**Q19. Statistical fixes to the acceptance criteria (initial thresholds may change only via the decision log).**
- **Default:** approve all of the following.
  - T1: gate on raw per-band rates at δ_j and on the family-wise rate at δ. Label vacuous cells as vacuous.
  - T1-expensive: allow violations consistent with δ per seed instead of zero tolerance.
  - T2: t-based, multiplicity-adjusted "significantly worse" rule. With 5 seeds and ~28 cells, the current rule false-fails about 80% of the time.
  - T4: seed stability judged by a confidence interval, not a raw 4-df standard deviation.
  - Rename "certified coverage" to "coverage at certified τ": only risk is certified.

---

## Tier 4: before A3 / A4

**Q20. ADBench subset.**
Healthcare is the largest single category among ADBench's 47 classical datasets (12 of 47).
- **Default:**
  - At most one dataset per ADBench category, and no category above one third of the subset.
  - n ≥ 3,000.
  - Base rates spread above and below the α grid.
  - Each dataset's category recorded in `NOTES.md`.

**Q21. Extraction units.**
Units defined from gold rows do not exist at deployment.
- **Default:**
  - Units are (document, schema field) pairs with "absent" as a value; spurious and missed fields score 1.
  - CORD is too small for group-level certification at tight bands, so use it at the 5% band only and add a larger set in wave 2.
  - CORD lives on Hugging Face and Google Drive, both blocked here, so it needs an allowlist or a copy.

**Q22. T3 evaluation population.**
- **Default:** report abstention recall (true positives that end up RESIDUAL) and detector AUROC within RESIDUAL, and gate on the latter. Keep CLINC150 out-of-scope queries out of dev for the `novel` test.
- T3 needs CLINC150 and Covertype, which are scheduled for A4, so their loaders move to A3. Covertype's hosts are blocked here.

**Q23. Domain vocabulary for T5.**
The 13 table mixes facets. "vision" is a modality, and BANKING77 and CLINC150 both cover banking.
- **Default:** use one single-facet domain list, with one domain per sub-dataset taken from the archive's own tags. A held-out domain is removed from every family. The T5 trajectory metric uses dev-simulated certified coverage with 3 paired seeds and a −0.02 non-inferiority margin.

**Q24. MCP server (O8).** **Default:** no.

**Q25. `amx doctor` naming.** This is the only remaining medical-sounding token. It is the standard CLI idiom (`brew doctor`, `flutter doctor`). **Default:** keep it, and allowlist it in the vocabulary check.

---

## Raised by the A0 build (2026-10-07)

Defaults D18 apply to G1 to G3 and Q1 to Q8; nothing above has been answered yet.

**Q26. Start rule and power at tight bands.**
The fixed-sequence walk starts where dev estimates reach 1.25 × n_min committed units (7.2 step 3). It is valid, but at tight bands it tests at the lowest-power point: on the synthetic oracle, generator (a) never certifies α ≤ 1% and certifies α = 5% in 57% of repetitions at n_calib = 10,000.
- (a) Keep the 1.25 × n_min rule.
- (b) Choose the start on dev by simulating the walk on calibration-sized bootstraps of dev OOF (still fixed before calibration, so still valid).
- **Default:** (a) in A0; compare (b) on T1 in A1 and adopt it if it raises certified coverage without changing the violation rates.

**Q27. The electricity gate series certifies nothing at the provisional Q5 settings.**
With relative tolerance 10% and bands {5%, 10%}, the A0 seasonal-naive baseline certifies no band. This is not a validity failure (no gross failure), but it leaves gate item 4 vacuous for forecasting. A pre-registered diagnostic on another client at bands {20%, 30%} (`datasets/electricity_client/diagnostic_b.yaml`) checks the forecasting path non-vacuously.
- (a) Keep the settings and let A1's forecasters raise coverage.
- (b) Loosen the bands.
- (c) Loosen the tolerance.
- **Default:** (a), with the diagnostic as evidence that the path works.

**Q28. Benchmark runs and the partition ledger.**
The A0 benchmark partitions (seed 20261007) were certified twice: once with the pre-review code (outputs deleted because of the slice leak) and once with the final code. The vault-wide partition ledger now refuses a second certification of a partition, so a future rerun of the A0 gate needs a fresh vault or a new benchmark seed.
- **Default:** benchmark seeds are disposable; release runs use their own seeds and are never re-split (A2 release mode).

---

## Mapping of the original O1 to O10

| Handoff | Here | Note |
|---|---|---|
| O1 license / name | G1, G3 | The repo is already public; PyPI `amx` is taken |
| O2 compute | Q9 | |
| O3 demo bands, δ | Q1, Q5 | δ accounting depends on Q1 |
| O4 forecasting dataset | Q10 | Both candidates blocked here |
| O5 backend | Q14 | |
| O6 agent budget | Q15 | Adds a program-level cap |
| O7 storage / PII | Q17 | Adds dev-data egress |
| O8 MCP | Q24 | |
| O9 vault mode | Q16 | No Docker daemon in cloud sessions |
| O10 encoders | Q9 | Hugging Face and PyTorch hosts blocked |

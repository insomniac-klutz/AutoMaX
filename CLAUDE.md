# AutoMaX (amx): rules for Claude Code

- Read HANDOFF.md sections 1-8 before changing anything. Sections 6 and 7 are normative, as amended by the corrections C1-C17 in docs/ROLLER.md.
- docs/ROLLER.md is the rolling build plan: status board, next action, file-by-file steps. Update it after every step.
- docs/OQ.md holds the owner's open questions. Do not decide them yourself; unanswered ones take their stated default when their milestone starts. Log every decision in HANDOFF.md section 17.2.
- Build in milestone order (HANDOFF section 14). Do not start the next milestone before its gate passes.
- NEVER read, list, grep or write anything under $AMX_VAULT, any path containing `vault/`,
  `folds/calib` or `folds/sealed`. Never print label columns of non-dev folds.
- During search sessions do not edit: src/amx/{cert,eval,split,warden,loss}/**,
  tests/{statistical,redteam}/**, claude-plugin/hooks/**, .claude/settings*.
  Changes there need a decision-log entry and owner approval.
- One change per experiment. Record the hypothesis and outcome with `amx ledger add`.
- LLM output never decides acceptance. Acceptance = `amx eval` JSON + the rule in program.md.
- Every statistical formula needs a Monte Carlo test. Label guarantee types truthfully.
- No domain strings (dataset names, column names, industries) in src/, claude-plugin/, programs/. Keep contract vocabulary domain-neutral (decision D17).
- Commands: `make check`, `make contract`, `make stat`, `make e2e-small`, `make redteam`.
- Style: ruff; mypy strict on spec/cert/loss; pydantic v2; logging, not print.
- Claude Code specifics change fast: verify hooks, skills, subagents and CLI flags against
  https://code.claude.com/docs before relying on them, and run `claude plugin validate`.

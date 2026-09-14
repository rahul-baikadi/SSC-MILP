# SSC-MILP — reimplementation

Independent reimplementation of **"MILP-Solve-Free Selection of LLM-Generated MILP
Programs via Multi-Signal Structural-Semantic Consistency"** (Zeng et al.).
The original repository link in the paper is dead; this code is written from the
paper alone.

SSC is a best-of-K selector over a fixed pool of LLM-generated Pyomo programs.
It never solves a candidate integer program: it combines structural consistency
(SC), static quality (SQ), text alignment (TA) and two bounded LP-relaxation
diagnostics (LP-sol, LP-obj).  SCIP is used only offline to label candidates.

## Layout

```
ssc/config.py         every published constant (Eqs. 8–9, 17–22, Table 8, Appendix A/L/M)
ssc/ast_checks.py     static defect detection (Table 8, Parse/AST + Pyomo-API rows)
ssc/sandbox_worker.py isolated construction + standard linear representation (no solver)
ssc/lp_worker.py      isolated GLPK relaxation (5 s limit + process watchdog)
ssc/scip_worker.py    isolated SCIP solve, OFFLINE EVALUATION ONLY (120 s)
ssc/sandbox.py        subprocess drivers for the three workers
ssc/signature.py      model signature M = (V, o, R), canonicalisation
ssc/similarity.py     sim(M_i, M_j): Eqs. 17–22
ssc/quality.py        SQ (Eq. 13) and the fatal cap
ssc/alignment.py      TA (Eq. 14): evidence features + closed-vocabulary coverage
ssc/lp_diag.py        LP-sol / LP-obj (Eqs. 15–16) with pool normalisation
ssc/selector.py       clone groups, SC (Eq. 23), fusion (Eqs. 8–9), ablation variants
ssc/templates.py      14 archetype templates + retrieval score (Eq. 24)
ssc/prompts.py        generation / repair / extraction / judge prompts
ssc/generate.py       K=10 generation, temperature schedule, repair loop, shuffle (App. M)
ssc/extract.py        semantic requirement extraction (App. L)
ssc/evaluate.py       SCIP labels (Eq. 1) and per-pool scoring
ssc/judge.py          zero-shot listwise LLM judge baseline (App. K)
ssc/metrics.py        accuracy / random / first / coverage / McNemar / Table 6–7 stats
scripts/run.py        CLI driver (generate | extract | score | evaluate | judge | report | all | check)
slurm/                CCR environment setup and batch script
data/benchmarks/      normalised NL4OPT (230) and MaMO-Easy (652) test files (LLMOPT release)
tests/                unit tests + offline end-to-end test with a mock LLM
```

## Running

```bash
python scripts/run.py check                       # solvers + API sanity check
python scripts/run.py all --benchmark nl4opt --judge
python scripts/run.py report --benchmark nl4opt   # runs/nl4opt/report.md
```

The DeepSeek key is read from `DEEPSEEK_API_KEY`, `DEEPSEEK_API_KEY_FILE`, or a
`deepseek_key` file in the repo or its parent.  All LLM calls are cached under
`cache/llm/` keyed by (model, messages, temperature, tag), so every stage resumes
and re-runs are free.  Model name defaults to `deepseek-v4-flash` (`--model` /
`SSC_LLM_MODEL` to override).

## What is transcribed vs. reconstructed

Transcribed exactly from the paper: all weights and thresholds (Eqs. 8–9, 17–23),
the defect taxonomy and penalties (Table 8), the fatal cap, LP diagnostics
(Eqs. 15–16, dim 30, GLPK 5 s), correctness tolerance (Eq. 1), K = 10, the five
formulation variants, the temperature schedule, two repair rounds at T = 0,
the per-problem shuffle seed `f"{seed}:{problem_id}"`, the 21-token TA
vocabulary, extraction settings (T = 0, 1400 tokens, 2 retries), the template
retrieval score (Eq. 24), the McNemar protocol.

Reconstructed (the paper describes them but does not publish them):

* **Prompt texts** — written to satisfy every stated contract (App. A, M.2–M.4,
  M.6, L, K).
* **Template library** — 14 archetypes written from textbook formulations.
* **Token vocabularies inside signatures** — variable-name tokens, magnitude
  bins (`floor(log10|x|)`), variable shape `arity:domain`, linear-term token
  `shape|sign+magnitude`, constraint expression profile.
* **AST rules** for each Table 8 defect (`ssc/ast_checks.py`).
* **Rule-based mapping** from constraint rows to the semantic constraint types
  of the TA vocabulary (`ssc/alignment.py::_row_features`).
* **Standardisation** uses Pyomo's `generate_standard_repn`; because Pyomo
  normalises relational expressions, the *original* sense of a row is taken from
  its normalised `lower/upper` form (a `>=` written as `a >= b` keeps `ge`).

Numbers will not match the paper: the candidate pool is a fresh stochastic
sample from the API.  The protocol and metrics are the same.

## Tests

```bash
python -m pytest tests -q
```

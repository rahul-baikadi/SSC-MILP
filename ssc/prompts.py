"""Prompt texts.  The paper publishes the *contracts* (Appendix A, M.2–M.4,
M.6, L) but not the verbatim prompts; these are written to satisfy every
stated requirement.  Do not add a reference answer anywhere."""
from __future__ import annotations

import json

from . import config as C
from .templates import render_template

CODE_CONTRACT = """\
Code contract (mandatory):
- Return a single fenced ```python code block containing a complete, runnable module.
- import pyomo.environ as pyo
- Put EVERY numeric constant from the problem in a module-level dictionary named DATA;
  the model must reference DATA entries only (no magic numbers in the model).
- Define def build_model(data=None): that uses `d = data or DATA`, creates
  model = pyo.ConcreteModel(), and RETURNS the model.
- All components are defined as model.<name> = pyo.Var / pyo.Constraint / pyo.Objective / pyo.Set / pyo.RangeSet.
- Exactly one objective, named model.obj = pyo.Objective(expr=..., sense=pyo.minimize or pyo.maximize).
- Variables must declare their domain explicitly: within=pyo.Binary / pyo.NonNegativeIntegers /
  pyo.Integers / pyo.NonNegativeReals / pyo.Reals, plus bounds=(lb, ub) when the text gives limits.
- Indexed constraints use named rule functions: def rule(m, i): ...; model.c = pyo.Constraint(model.I, rule=rule).
  Do NOT use pyo.ConstraintList, .add(), lambda rules, @model.Constraint decorators, setattr, or
  add_component.  Do not name a component like a Block attribute (name, model, index, type, display, ...).
- The model must be LINEAR: never multiply or divide two decision variables, never use pyo.max/pyo.min,
  abs(), or Python max()/min() on expressions.  Linearise if-then logic with binary variables and big-M.
- Do NOT create a solver, call SolverFactory, .solve(), or pyo.value() anywhere in the module.
- If you need a placeholder define GOLD_ANSWER = None; never include, infer, or use a reference answer.
"""

COT_STEPS = """\
Before writing code, reason through these six steps explicitly:
1. Sets and indices: list every collection of entities (machines, products, time periods, ...).
2. Variables: for each decision, name the variable, its index set, and its domain (Binary / Integer / NonNegativeReals / Reals).
3. Objective: write the objective expression in words, then confirm minimize vs. maximize.
4. Constraints: enumerate every restriction in the problem text; check that each sentence maps to at least one constraint.
5. Data usage: confirm every numeric value in the problem appears in DATA and is referenced in the model.
6. Linearity: ensure no products of two decision variables appear in any expression.
"""

VARIANTS = [
    ("natural_naming",
     "Formulation emphasis: use descriptive variable names (e.g. production[i], assign[i,j]); "
     "define one Pyomo Constraint component per semantic restriction in the text."),
    ("data_first",
     "Formulation emphasis: before defining any Pyomo component, enumerate every numeric value from the "
     "problem in DATA together with a comment quoting its source sentence; reference only DATA entries in the model."),
    ("compact_index",
     "Formulation emphasis: use short variable names (x, y, z) indexed by explicit pyo.RangeSet / pyo.Set; "
     "after writing each constraint, verify its direction (<= vs >=) against the problem text in a comment."),
    ("decomposed",
     "Formulation emphasis: write one Pyomo Constraint rule per atomic sentence of the problem; introduce "
     "auxiliary binary variables for any if-then logic or indicator condition and linearise it with big-M."),
    ("audit_first",
     "Formulation emphasis: begin the module with a structured comment block listing all decision variables "
     "(with index sets and domains), the objective in words, and each constraint numbered with its source "
     "sentence; then translate directly to Pyomo and verify every sense/direction a second time."),
]


def system_prompt() -> str:
    return ("You are an expert operations-research modeller. Translate natural-language optimisation "
            "problems into executable Pyomo MILP/LP models.\n\n" + CODE_CONTRACT)


def generation_messages(problem: str, templates: list[dict], variant_idx: int) -> list[dict]:
    vname, vtext = VARIANTS[variant_idx % len(VARIANTS)]
    tmpl = "\n".join(render_template(t) for t in templates)
    tmpl_block = (f"## Retrieved modelling templates (reference only; adapt, do not copy blindly)\n{tmpl}\n"
                  if templates else "")
    user = (f"## Problem\n{problem.strip()}\n\n"
            f"{tmpl_block}"
            f"## Reasoning steps\n{COT_STEPS}\n"
            f"## {vtext}\n\n"
            "Work through the six reasoning steps briefly, then output the complete module in one "
            "```python block that satisfies the code contract.")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def repair_messages(problem: str, templates: list[dict], variant_idx: int, prior_code: str,
                    feedback: dict) -> list[dict]:
    vname, vtext = VARIANTS[variant_idx % len(VARIANTS)]
    tmpl = "\n".join(render_template(t) for t in templates)
    tmpl_block = f"## Retrieved modelling templates\n{tmpl}\n" if templates else ""
    fb = json.dumps(feedback, indent=1)[:4000]
    user = (f"## Problem\n{problem.strip()}\n\n"
            f"{tmpl_block}"
            f"## {vtext}\n\n"
            f"## Previous attempt\n```python\n{prior_code}\n```\n\n"
            f"## Structured feedback from sandbox and static checks\n{fb}\n\n"
            "Repair the module so that it constructs without error and satisfies every item of the code "
            "contract, while preserving the mathematical intent of the original formulation. Do not solve "
            "the model. Output the complete corrected module in one ```python block.")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- Appendix L
def extraction_messages(problem: str) -> list[dict]:
    vocab = "\n".join(f"- {t}" for t in C.TA_VOCABULARY)
    sys = ("You extract structural requirements from optimisation problem statements. You never solve the "
           "problem, never compute or guess its answer, and never write code.")
    user = (f"## Problem\n{problem.strip()}\n\n"
            "## Task\nList the semantic requirements that any correct mathematical model of this problem must "
            "satisfy, using ONLY the closed feature vocabulary below. Return a JSON object of the form\n"
            '{"requirements": [{"name": "<short identifier>", "features": ["<token>", ...], "weight": <0.1-2.0>}, ...]}\n'
            "Guidance: objective sense is typically weight 1.5; structural constraint types (assignment, flow "
            "balance, capacity, ...) 1.0-1.5; simple bound-type constraints 0.5-1.0; variable domain facts "
            "(binary / integer / bounded) 0.5-1.0. Include a requirement only if the text clearly implies it. "
            "Use one feature token per requirement unless two tokens are jointly required.\n\n"
            f"## Closed vocabulary\n{vocab}\n\nReturn only the JSON object.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- Appendix K
def judge_messages(problem: str, codes: list[str]) -> list[dict]:
    sys = ("You are a careful reviewer of optimisation models. You are shown a problem and several candidate "
           "Pyomo programs. You cannot run code and no reference answer is available.")
    body = "\n\n".join(f"### Candidate {i}\n```python\n{c}\n```" for i, c in enumerate(codes))
    user = (f"## Problem\n{problem.strip()}\n\n## Candidates\n{body}\n\n"
            "Select the single candidate whose model most faithfully and correctly formulates the problem. "
            'Answer with a JSON object {"choice": <candidate index>, "reason": "<one sentence>"} and nothing else.')
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

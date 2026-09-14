"""Deterministic mock LLM for offline end-to-end tests.  It answers generation
prompts with hand-written Pyomo programs (correct / wrong / broken variants
chosen by variant index and temperature), extraction prompts with a fixed
requirement list, and judge prompts with candidate 0."""
from __future__ import annotations

import json
import re
from pathlib import Path

FIX = Path(__file__).resolve().parent / "fixtures"

DUCKS = (FIX / "ducks_ok.py").read_text()
PILLS = """\
import pyomo.environ as pyo

DATA = {"ingredients": 1000, "large_ing": 3, "small_ing": 2, "large_fill": 2, "small_fill": 1,
        "min_large": 100, "small_frac": 0.6}

def build_model(data=None):
    d = data or DATA
    model = pyo.ConcreteModel()
    model.large = pyo.Var(within=pyo.NonNegativeIntegers)
    model.small = pyo.Var(within=pyo.NonNegativeIntegers)
    model.obj = pyo.Objective(expr=d["large_fill"] * model.large + d["small_fill"] * model.small, sense=pyo.minimize)
    def ing_rule(m):
        return d["large_ing"] * m.large + d["small_ing"] * m.small <= d["ingredients"]
    model.ing = pyo.Constraint(rule=ing_rule)
    def min_large_rule(m):
        return m.large >= d["min_large"]
    model.min_large = pyo.Constraint(rule=min_large_rule)
    def frac_rule(m):
        return m.small >= d["small_frac"] * (m.large + m.small)
    model.frac = pyo.Constraint(rule=frac_rule)
    return model
"""

PROBLEMS = [
    {"pid": "toy_0001", "index": 1, "answer": 1160.0, "answer_raw": 1160,
     "question": ("There has been an oil spill in the ocean and ducks need to be taken to shore to be cleaned "
                  "either by boat or by canoe. A boat can take 10 ducks per trip while a canoe can take 8 ducks "
                  "per trip. Since the boats are motor powered, they take 20 minutes per trip while the canoes "
                  "take 40 minutes per trip. In order to avoid further environmental damage, there can be at most "
                  "12 boat trips and at least 60% of the trips should be by canoe. If at least 300 ducks need to "
                  "be taken to shore, how many of each transportation method should be used to minimize the "
                  "total amount of time needed to transport the ducks?")},
    {"pid": "toy_0002", "index": 2, "answer": 350.0, "answer_raw": 350,
     "question": ("A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. "
                  "A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill "
                  "requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least "
                  "100 large pills. However, since small pills are more popular at least 60% of the total number "
                  "of pills must be small. How many of each should be made to minimize the total number of filler "
                  "material needed?")},
]


def _variants(base: str) -> dict[str, str]:
    return {
        "ok": base,
        "clone": re.sub(r"\bmodel\.(\w+)", lambda m: "model." + m.group(1).upper() if m.group(1) not in ("obj",) else m.group(0), base)
                 .replace("m.boats", "m.BOATS").replace("m.canoes", "m.CANOES")
                 .replace("m.large", "m.LARGE").replace("m.small", "m.SMALL"),
        "wrong_sense": base.replace("sense=pyo.minimize", "sense=pyo.maximize"),
        "wrong_data": base.replace('"boat_cap": 10', '"boat_cap": 100').replace('"large_ing": 3', '"large_ing": 30'),
        "syntax": base.replace("return model", "return model)"),
        "nonlinear": base.replace("* model.boats +", "* model.boats * model.canoes +")
                         .replace("* model.large +", "* model.large * model.small +"),
        "solver": base.replace("    return model", '    pyo.SolverFactory("glpk").solve(model)\n    return model'),
        "clist": base.replace("model.ducks = pyo.Constraint(rule=ducks_rule)",
                              "model.cl = pyo.ConstraintList()\n    model.cl.add(10 * model.boats + 8 * model.canoes >= 300)")
                     .replace("model.ing = pyo.Constraint(rule=ing_rule)",
                              "model.cl = pyo.ConstraintList()\n    model.cl.add(3 * model.large + 2 * model.small <= 1000)"),
        "noobj": re.sub(r"\n    model\.obj = .*", "", base),
        "continuous": base.replace("NonNegativeIntegers", "NonNegativeReals"),
    }


# recipe (variant, repeat) -> which program the "LLM" writes
_TABLE = {(0, 0): "ok", (1, 0): "clone", (2, 0): "wrong_sense", (3, 0): "syntax", (4, 0): "nonlinear",
          (0, 1): "ok", (1, 1): "solver", (2, 1): "clist", (3, 1): "wrong_data", (4, 1): "continuous"}

_REQS = {"requirements": [
    {"name": "minimize", "features": ["objective:minimize"], "weight": 1.5},
    {"name": "integer counts", "features": ["var:integer"], "weight": 1.0},
    {"name": "demand", "features": ["constraint:demand"], "weight": 1.0},
    {"name": "upper bound", "features": ["constraint:upper_bound"], "weight": 0.5},
    {"name": "bogus", "features": ["not:a_token"], "weight": 1.0},
]}


def mock_llm(messages, temperature, tag=""):
    user = messages[-1]["content"]
    if "Closed vocabulary" in user:
        return json.dumps(_REQS)
    if "Select the single candidate" in user:
        return '{"choice": 0, "reason": "mock"}'
    base = DUCKS if "ducks" in user else PILLS
    v = _variants(base)
    if "## Previous attempt" in user:
        # repair: return the correct program unless the prior was the solver
        # variant, which the mock stubbornly keeps to exercise the fatal cap
        if "SolverFactory" in user.split("## Previous attempt")[1]:
            return "```python\n" + v["solver"] + "```"
        return "```python\n" + v["ok"] + "```"
    m = re.search(r":v(\d):r(\d)", tag)
    idx, rep = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return "Reasoning...\n```python\n" + v[_TABLE[(idx, rep)]] + "```"

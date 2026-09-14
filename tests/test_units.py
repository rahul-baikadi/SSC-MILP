import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ssc.ast_checks import static_check  # noqa: E402
from ssc.lp_diag import lp_diagnostics, profile  # noqa: E402
from ssc.quality import has_fatal, static_quality  # noqa: E402
from ssc.selector import clone_groups, structural_consistency  # noqa: E402
from ssc.similarity import bag_jaccard, h, jaccard  # noqa: E402
from ssc.templates import retrieve_templates  # noqa: E402
from ssc.generate import extract_code, shuffled_positions  # noqa: E402


def test_jaccard():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1}, set()) == 0.0
    assert bag_jaccard(Counter(a=2, b=1), Counter(a=1, c=1)) == 1 / 4
    assert h(10, 10) == 1.0 and h(0, 0) == 1.0 and abs(h(10, 5) - 0.5) < 1e-12


def test_static_quality():
    assert static_quality([]) == 1.0
    assert static_quality(["syntax_error"]) == 0.0
    assert abs(static_quality(["possible_nonlinearity", "unknown_variable_domain"]) - 0.77) < 1e-12
    assert has_fatal(["disallowed_constraint_list"]) and not has_fatal(["possible_nonlinearity"])


def test_static_checks():
    code = (ROOT / "tests" / "fixtures" / "ducks_ok.py").read_text()
    assert static_check(code)["defects"] == []
    bad = code.replace("rule=max_boats_rule", "rule=lambda m: m.boats <= 12")
    assert "disallowed_lambda_rule" in static_check(bad)["defects"]
    bad = code.replace("return model", "pyo.SolverFactory('glpk').solve(model)\n    return model")
    assert "contains_solver_call" in static_check(bad)["defects"]
    bad = code.replace("model.boats = pyo.Var(within=pyo.NonNegativeIntegers)", "model.name = pyo.Var(within=pyo.NonNegativeIntegers)")
    assert "reserved_pyomo_component_name" in static_check(bad)["defects"]
    assert static_check("def build_model(:\n  pass")["defects"] == ["syntax_error"]


def test_lp_diag():
    res = [{"status": "optimal", "objective": 10.0, "values": [1, 2, 3]},
           {"status": "optimal", "objective": 10.0, "values": [3, 2, 1]},
           {"status": "optimal", "objective": 100.0, "values": [0, 0, 9]},
           {"status": "infeasible", "objective": None, "values": []}]
    sol, obj = lp_diagnostics(res)
    assert sol[0] == sol[1] == 1.0 and sol[3] == 0.0 and 0 < sol[2] < 1
    assert obj[0] == obj[1] == 1.0 and obj[2] == 0.0 and obj[3] == 0.0
    assert len(profile([5, 1, 3])) == 30 and list(profile([5, 1, 3])[:3]) == [1, 3, 5]
    # single feasible candidate: R < eps -> R = 1, score 1
    _, obj = lp_diagnostics([{"status": "optimal", "objective": 4.0, "values": [1]}])
    assert obj == [1.0]


def test_clone_weighting():
    S = [[1, 0.99, 0.5], [0.99, 1, 0.5], [0.5, 0.5, 1]]
    g = clone_groups(S)
    assert g[0] == g[1] != g[2]
    SC = structural_consistency(S, [1, 1, 1], [1, 1, 1], g)
    # candidate 2's peers 0 and 1 each get weight 1/2
    assert abs(SC[2] - 0.5) < 1e-12
    assert abs(SC[0] - (0.5 * 0.99 + 1 * 0.5) / 1.5) < 1e-12


def test_templates_and_shuffle():
    t = retrieve_templates("Ship goods from three warehouses to four stores at minimum shipping cost given supply and demand.")
    assert t[0]["name"] == "Transportation"
    assert retrieve_templates("xyzzy")[0]["name"] == "General LP/MILP"
    p = shuffled_positions("nl4opt_0001")
    assert sorted(p) == list(range(10)) and p == shuffled_positions("nl4opt_0001")
    assert extract_code("text\n```python\nx=1\n```\nmore ```py\ndef build_model():\n  pass\n```") == "def build_model():\n  pass\n"

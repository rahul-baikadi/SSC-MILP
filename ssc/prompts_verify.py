"""Pointwise, solution-aware verifier prompt (one call per candidate).
The verifier never sees the gold answer, SSC scores, or other candidates."""
from __future__ import annotations

VERIFIER_SYSTEM = (
    "You are an expert operations-research reviewer. You audit ONE optimisation model written in "
    "Pyomo for ONE problem and decide whether it faithfully represents that problem. No reference "
    "answer is available and you must not compute the optimum yourself. Judge only from the problem "
    "text, the code, and the solver result shown. Be strict about real errors and do not reject a "
    "model for style, naming, redundant-but-harmless constraints, or an equivalent formulation.")

CHECKLIST = """## Audit checklist (work through every item)
1. Decision variables: does each quantity in the problem have a variable with the right meaning and
   index set? Is each domain right? Quantities of indivisible things (people, vehicles, machines,
   servings, trips, items) should be integers, and yes/no choices binary, unless the text allows
   fractions.
2. Objective: correct sense (min/max), correct quantity, every cost/profit term with the right
   coefficient from the text.
3. Constraints: for each limit or requirement stated in the text, is it present, with the right
   direction (<=, >=, =), the right right-hand side, and the right coefficients? Is any constraint
   present that the text does not support?
4. Data and units: are numbers copied correctly, are units/percentages/rates converted correctly,
   and does the objective measure what the question asks for?
5. Solution check: substitute the variable values into the stated limits. Do they hold? Do the
   values make sense for the real-world objects (e.g. fractional people or vehicles)? If the status
   is not optimal, decide whether the problem as written should be feasible.

## Output
Return ONLY this JSON object:
{
  "verdict": "correct" | "incorrect",
  "p_correct": <number in [0,1]: probability that this model's optimal value is the answer the problem intends>,
  "errors": [<short descriptions of issues that contradict the problem text>],
  "assumptions": [<choices the text leaves open that this model made, e.g. "servings treated as continuous">],
  "reason": "<one sentence>"
}
Mark "incorrect" if any item in "errors" would change the optimal value. Items that are only
"assumptions" lower p_correct but do not by themselves make the model incorrect."""


def solver_block(sol: dict, max_vars: int = 40) -> str:
    lines = [f"status: {sol.get('status')}",
             f"objective value: {sol.get('objective') if sol.get('objective') is not None else 'n/a'}"]
    vs = sol.get("vars") or {}
    if vs:
        items = list(vs.items())
        lines.append("non-zero decision variables:")
        lines += [f"{k} = {v:.6g}" for k, v in items[:max_vars]]
        if len(items) > max_vars:
            lines.append(f"... ({len(items) - max_vars} more non-zero variables not shown)")
    elif sol.get("status") == "optimal":
        lines.append("non-zero decision variables: none (all zero)")
    return "\n".join(lines)


# Fixed instructions live in the system prompt so that system + problem form a cacheable prefix
SYSTEM_FULL = VERIFIER_SYSTEM + "\n\n" + CHECKLIST


def verifier_blocks(problem: str, code: str, sol: dict) -> list[dict]:
    """Problem block is cached (shared by the problem's 10 candidates); the candidate block varies."""
    return [
        {"text": f"## Problem\n{problem.strip()}", "cache": True},
        {"text": f"## Candidate model\n```python\n{code}\n```\n\n## Solver result for this model\n"
                 f"{solver_block(sol)}\n\nWork through the audit checklist, then return only the JSON object.",
         "cache": False},
    ]

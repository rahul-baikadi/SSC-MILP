"""Round-2 (diagnosis-guided regeneration) prompts.

Two separate calls per problem:

  Stage A  decision_messages()   one call, JSON out: the modelling decisions the
                                 text does not force, with options, plausibility
                                 and the option the round-1 majority used.
  Stage B  branch_messages()     one call per decision assignment, code out.

Plus the "feedback" control arm (diagnostics + diff, no explicit decisions):
  feedback_messages()

Every slot is filled from gold-free round-1 artifacts (LP-relaxation clusters,
model signatures, TA requirements, SQ defects).  The single few-shot example is
hand-written and unrelated to every benchmark; it shows the FORMAT only.
"""
from __future__ import annotations

import json

from .prompts import CODE_CONTRACT, system_prompt

# --------------------------------------------------------------------- example
# Hand-written toy problem (a bakery), not drawn from any evaluation set.
EXAMPLE_PROBLEM = (
    "A bakery makes loaves and cakes. A loaf uses 0.5 kg of flour and a cake uses 0.8 kg. "
    "At most 40 kg of flour is available per day. The bakery must make at least 20 items in total. "
    "Each loaf earns $3 and each cake earns $7. How many of each should be made per day?")

EXAMPLE_OUTPUT = {
    "decisions": [
        {"decision": "domain_of_item_counts",
         "evidence": "none (the text never says whole items)",
         "options": ["NonNegativeIntegers", "NonNegativeReals"],
         "plausibility": [0.5, 0.5],
         "majority_option": "NonNegativeReals"},
        {"decision": "objective_sense",
         "evidence": "\"Each loaf earns $3 and each cake earns $7\"",
         "options": ["maximize revenue", "minimize cost"],
         "plausibility": [0.95, 0.05],
         "majority_option": "maximize revenue"},
        {"decision": "total_items_direction",
         "evidence": "\"must make at least 20 items in total\"",
         "options": [">= 20", "<= 20"],
         "plausibility": [0.97, 0.03],
         "majority_option": ">= 20"},
    ]
}

# ------------------------------------------------------------------- stage A
STAGE_A_SYSTEM = (
    "You are an expert operations-research modeller reviewing alternative formulations of one "
    "optimisation problem. You never solve the problem, never compute or guess its optimal value, "
    "and never write code in this step.")

STAGE_A_TASK = """\
## Task
List every modelling decision that the problem text does NOT force uniquely. Always consider:
  - the domain of each decision variable (Binary / NonNegativeIntegers / NonNegativeReals / Reals),
  - the objective sense and what exactly is being optimised,
  - the direction (<=, >=, =) and inclusiveness of each limit,
  - the units / scale of the quantity the question asks for,
  - the orientation of tabular data (which index is the row, which the column),
  - constraints that are implied but not stated, or stated but possibly redundant.
Include every difference and check listed above. Omit decisions the text settles beyond doubt.

For each decision return:
  "decision":        short snake_case name
  "evidence":        the exact quoted sentence(s) that bear on it, or "none"
  "options":         2 or 3 mutually exclusive alternatives, each concrete enough to implement
  "plausibility":    one probability per option, summing to 1, from the text alone
  "majority_option": the option most existing models use (copy one of "options" exactly),
                     or null if the models above do not reveal it

Return a JSON object {"decisions": [...]} with at most 6 decisions, most uncertain first."""


def decision_messages(problem: str, slots: dict) -> list[dict]:
    reps = "".join(f"### {name}\n```python\n{code}\n```\n" for name, code in slots["representatives"])
    user = (
        f"## Problem\n{problem.strip()}\n\n"
        f"## Existing candidate models (no reference answer is known)\n{slots['pool_summary']}\n\n"
        f"## Representative models (highest-scored model of each group)\n{reps}\n"
        f"## Where the existing models differ\n{slots['signature_diff']}\n\n"
        f"## Automatic checks\n{slots['diagnostics']}\n\n"
        f"{STAGE_A_TASK}\n\n"
        "## Example of the required format (unrelated problem; do not reuse its content)\n"
        f"Problem: {EXAMPLE_PROBLEM}\n"
        f"Output:\n{json.dumps(EXAMPLE_OUTPUT, indent=1)}\n\n"
        "Return only the JSON object.")
    return [{"role": "system", "content": STAGE_A_SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------------- stage B
def branch_messages(problem: str, assignment: list[dict]) -> list[dict]:
    """assignment: [{"decision", "option", "evidence"}, ...]"""
    lines = []
    for a in assignment:
        tag = "text silent; treat as an assumption" if a.get("evidence", "none").startswith("none") \
            else f"source: {a['evidence']}"
        lines.append(f"- {a['decision']}: {a['option']}   ({tag})")
    user = (
        f"## Problem\n{problem.strip()}\n\n"
        "## Modelling decisions to follow (fixed; do not revisit them)\n" + "\n".join(lines) + "\n\n"
        "## Task\n"
        "Write the complete Pyomo model that implements every decision above exactly. All other "
        "modelling choices must follow the problem text. Next to each variable, the objective and each "
        "constraint, add a comment quoting the sentence that justifies it, or "
        "'# assumption: <decision>' when it comes from a decision above that the text does not state. "
        "Do not solve the model.\n"
        "Output the module in one ```python block that satisfies the code contract.")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


# ------------------------------------------------------- feedback control arm
def feedback_messages(problem: str, slots: dict) -> list[dict]:
    reps = "".join(f"### {name}\n```python\n{code}\n```\n" for name, code in slots["representatives"])
    user = (
        f"## Problem\n{problem.strip()}\n\n"
        f"## Existing candidate models (no reference answer is known)\n{slots['pool_summary']}\n\n"
        f"## Representative models\n{reps}\n"
        f"## Structural differences between the groups\n{slots['signature_diff']}\n\n"
        f"## Checks that need verification\n{slots['diagnostics']}\n\n"
        "## Task\n"
        "For EACH listed difference and check: (1) quote the sentence(s) that decide it; (2) state which "
        "choice the text supports, or 'not determined by the text'. Then write one corrected model that "
        "follows these decisions. Do not copy a group's model where a decision contradicts it. "
        "Do not solve the model.\n"
        "Output the module in one ```python block that satisfies the code contract.")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


__all__ = ["decision_messages", "branch_messages", "feedback_messages", "CODE_CONTRACT"]

"""Template library and retrieval (Appendix M.1).

The paper describes 14 canonical MILP archetypes, each recording: problem type
name, natural-language description, canonical formulation sketch, reasoning
flow steps, and common modelling errors.  The template *texts* are not
published; the ones below are written from standard OR textbook formulations.
The retrieval score is Eq. 24, transcribed exactly.
"""
from __future__ import annotations

import re

from . import config as C

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


TEMPLATES: list[dict] = [
    {
        "name": "Transportation",
        "keywords": ["transportation", "shipping cost", "ship from", "supply", "demand", "warehouse",
                     "source", "destination", "distribution center", "shipped", "transport"],
        "description": "Ship goods from supply nodes to demand nodes at minimum total shipping cost.",
        "formulation": (
            "Sets: sources I, destinations J.  Data: supply s_i, demand d_j, unit cost c_ij.\n"
            "Vars: x[i,j] >= 0 (units shipped, integer if indivisible).\n"
            "min sum_ij c_ij x_ij\n"
            "s.t. sum_j x_ij <= s_i  (supply)   for all i\n"
            "     sum_i x_ij >= d_j  (demand)   for all j"),
        "reasoning": ["identify sources and destinations", "collect supplies, demands, unit costs",
                      "decide whether supply is <= or = and demand is >= or =",
                      "check total supply >= total demand"],
        "errors": ["swapping supply and demand directions", "using = when the text says 'at most'",
                   "forgetting non-negativity"],
    },
    {
        "name": "Assignment",
        "keywords": ["assign", "assigned", "assignment", "each worker", "each task", "each job",
                     "exactly one", "one machine", "matching"],
        "description": "Assign agents to tasks one-to-one (or with capacities) at minimum cost / maximum value.",
        "formulation": (
            "Sets: agents I, tasks J.  Data: cost/value c_ij.\n"
            "Vars: x[i,j] in {0,1}.\n"
            "min/max sum_ij c_ij x_ij\n"
            "s.t. sum_j x_ij = 1 (or <= 1) for all i;  sum_i x_ij = 1 (or >= 1) for all j"),
        "reasoning": ["identify the two sides being matched", "decide = vs <= vs >= for each side",
                      "binary variables per pair"],
        "errors": ["using continuous variables when the text needs whole assignments",
                   "wrong equality/inequality on one side"],
    },
    {
        "name": "Knapsack / Project Selection",
        "keywords": ["knapsack", "select", "selected", "choose", "chosen", "budget", "weight limit",
                     "capacity", "value", "profit", "invest", "projects", "items"],
        "description": "Choose a subset of items maximizing total value subject to one or more capacity/budget limits.",
        "formulation": (
            "Set: items I.  Data: value v_i, weight/cost w_i, capacity W.\n"
            "Vars: x[i] in {0,1} (or integer counts).\n"
            "max sum_i v_i x_i   s.t.  sum_i w_i x_i <= W"),
        "reasoning": ["list items with value and resource use", "identify every resource limit",
                      "binary (select) vs integer (how many)"],
        "errors": ["treating a count variable as binary", "missing a second resource constraint"],
    },
    {
        "name": "Facility Location",
        "keywords": ["facility", "facilities", "open", "opening", "fixed cost", "warehouse", "plant",
                     "location", "locations", "site", "sites", "serve customers", "depot"],
        "description": "Decide which facilities to open and how to serve customers, minimizing fixed plus service cost.",
        "formulation": (
            "Sets: sites I, customers J.  Data: fixed cost f_i, capacity u_i, demand d_j, cost c_ij.\n"
            "Vars: y[i] in {0,1} (open), x[i,j] >= 0 (flow or fraction).\n"
            "min sum_i f_i y_i + sum_ij c_ij x_ij\n"
            "s.t. sum_i x_ij = d_j (or =1 if fractions);  sum_j x_ij <= u_i y_i  (linking)"),
        "reasoning": ["separate open/close decisions from service decisions",
                      "write the linking constraint with the capacity or big-M",
                      "check demand satisfaction"],
        "errors": ["forgetting the linking constraint", "big-M too small"],
    },
    {
        "name": "Network Flow",
        "keywords": ["flow", "network", "arc", "arcs", "edge", "edges", "node", "nodes", "path",
                     "pipeline", "conservation", "sent through", "max flow", "shortest path"],
        "description": "Route flow through a network respecting arc capacities and flow conservation at nodes.",
        "formulation": (
            "Sets: nodes N, arcs A.  Data: capacity u_a, cost c_a, node supply b_n.\n"
            "Vars: f[a] >= 0.\n"
            "min sum_a c_a f_a\n"
            "s.t. sum_{a out of n} f_a - sum_{a into n} f_a = b_n  for all n;   0 <= f_a <= u_a"),
        "reasoning": ["list nodes and directed arcs", "write conservation at every node",
                      "sign convention: out minus in equals supply"],
        "errors": ["conservation at source/sink written wrong", "undirected arcs modelled as one variable"],
    },
    {
        "name": "Scheduling",
        "keywords": ["schedule", "scheduling", "makespan", "job", "jobs", "operation", "operations",
                     "machine", "machines", "processing time", "deadline", "due date", "start time",
                     "completion time", "precedence"],
        "description": "Sequence jobs on machines with precedence / non-overlap constraints, minimizing makespan or lateness.",
        "formulation": (
            "Sets: jobs J, machines M.  Data: processing time p_jm, big-M.\n"
            "Vars: s[j] >= 0 (start), C_max >= 0, y[j,k] in {0,1} (j before k).\n"
            "min C_max\n"
            "s.t. s_k >= s_j + p_j - M (1 - y_jk);  s_j >= s_k + p_k - M y_jk;  C_max >= s_j + p_j"),
        "reasoning": ["identify jobs, machines, processing times", "precedence within a job",
                      "disjunctive (either/or) constraints via binaries and big-M",
                      "define the makespan variable"],
        "errors": ["nonlinear products of start times and binaries", "missing either/or pair"],
    },
    {
        "name": "Production Planning / Product Mix",
        "keywords": ["produce", "production", "product", "products", "manufacture", "units of",
                     "per unit", "hours", "labor", "labour", "raw material", "machine time", "profit",
                     "make", "made"],
        "description": "Choose production quantities of several products to maximize profit under resource limits.",
        "formulation": (
            "Set: products P, resources R.  Data: profit c_p, usage a_rp, availability b_r.\n"
            "Vars: x[p] >= 0 (integer if units are indivisible).\n"
            "max sum_p c_p x_p   s.t.  sum_p a_rp x_p <= b_r  for all r;  min/max demand bounds"),
        "reasoning": ["list products and their per-unit resource usage", "one constraint per resource",
                      "check whether quantities must be integers", "add minimum/maximum demand bounds"],
        "errors": ["missing a resource constraint mentioned in prose", "percentage constraints written nonlinearly"],
    },
    {
        "name": "Blending / Diet",
        "keywords": ["blend", "blending", "diet", "mix", "mixture", "nutrient", "nutrients", "calories",
                     "protein", "vitamin", "ingredient", "ingredients", "at least", "at most", "percent",
                     "%", "proportion", "ratio"],
        "description": "Combine ingredients to meet nutritional / composition requirements at minimum cost.",
        "formulation": (
            "Set: ingredients I, nutrients N.  Data: cost c_i, content a_ni, requirement r_n (min/max).\n"
            "Vars: x[i] >= 0.\n"
            "min sum_i c_i x_i   s.t.  sum_i a_ni x_i >= r_n (or <=);  ratio constraints linearised"),
        "reasoning": ["identify ingredients and per-unit contents", "one constraint per requirement",
                      "convert percentage / ratio statements into linear form: x_a >= 0.6 (x_a + x_b)"],
        "errors": ["dividing by a variable", "wrong direction on 'at least / at most'"],
    },
    {
        "name": "Bin Packing / Cutting Stock",
        "keywords": ["bin", "bins", "pack", "packing", "container", "containers", "truck", "trucks",
                     "box", "boxes", "cut", "cutting", "roll", "rolls", "pattern", "fit"],
        "description": "Pack items into the fewest bins (or cut stock with least waste) without exceeding capacity.",
        "formulation": (
            "Sets: items I, bins B.  Data: size s_i, capacity C.\n"
            "Vars: x[i,b] in {0,1}, y[b] in {0,1}.\n"
            "min sum_b y_b   s.t.  sum_b x_ib = 1;  sum_i s_i x_ib <= C y_b"),
        "reasoning": ["upper-bound the number of bins", "each item in exactly one bin",
                      "capacity linking with the bin-used binary"],
        "errors": ["capacity constraint not linked to y", "symmetric solutions slowing solves (harmless)"],
    },
    {
        "name": "Set Covering / Location Covering",
        "keywords": ["cover", "covered", "covering", "coverage", "station", "stations", "reach",
                     "within", "minutes", "miles", "km", "at least one", "service area", "ambulance",
                     "fire"],
        "description": "Choose the fewest (or cheapest) facilities so that every demand point is covered.",
        "formulation": (
            "Sets: sites I, points J.  Data: coverage a_ij in {0,1}, cost c_i.\n"
            "Vars: y[i] in {0,1}.\n"
            "min sum_i c_i y_i   s.t.  sum_i a_ij y_i >= 1  for all j"),
        "reasoning": ["build the coverage matrix from distances / lists",
                      "one >= 1 constraint per demand point"],
        "errors": ["using = 1 instead of >= 1", "forgetting some demand points"],
    },
    {
        "name": "Inventory / Lot Sizing",
        "keywords": ["inventory", "period", "periods", "month", "months", "week", "weeks", "quarter",
                     "holding cost", "storage", "stock", "backlog", "setup cost", "carry", "carried over",
                     "beginning", "ending"],
        "description": "Plan production and inventory over periods to meet demand at minimum production, setup and holding cost.",
        "formulation": (
            "Set: periods T.  Data: demand d_t, unit cost c_t, holding h_t, setup f_t, capacity u_t.\n"
            "Vars: x[t] >= 0, I[t] >= 0, y[t] in {0,1}.\n"
            "min sum_t (c_t x_t + h_t I_t + f_t y_t)\n"
            "s.t. I_{t-1} + x_t - I_t = d_t;  x_t <= u_t y_t;  I_0 given"),
        "reasoning": ["write the inventory balance for every period", "handle the initial inventory",
                      "setup binaries if fixed costs exist"],
        "errors": ["off-by-one in the balance equation", "ignoring initial/final inventory"],
    },
    {
        "name": "Vehicle Routing / TSP",
        "keywords": ["route", "routes", "routing", "tour", "visit", "visits", "vehicle", "vehicles",
                     "travel", "distance", "cities", "city", "customers", "return", "salesman"],
        "description": "Find minimum-length tours visiting every customer once, subject to vehicle capacity.",
        "formulation": (
            "Set: nodes N (0 = depot), arcs (i,j).  Data: distance d_ij, capacity Q, demand q_i.\n"
            "Vars: x[i,j] in {0,1}, u[i] >= 0 (MTZ load / order).\n"
            "min sum_ij d_ij x_ij\n"
            "s.t. sum_j x_ij = 1, sum_i x_ij = 1 (degree);  u_j >= u_i + q_j - Q (1 - x_ij) (subtour / capacity)"),
        "reasoning": ["degree constraints in and out", "subtour elimination (MTZ) with continuous order variables",
                      "no x_ii arcs"],
        "errors": ["missing subtour elimination", "nonlinear subtour constraints"],
    },
    {
        "name": "Workforce / Shift Scheduling",
        "keywords": ["shift", "shifts", "employees", "employee", "workers", "staff", "nurses", "staffing",
                     "days", "day", "hours per", "part time", "full time", "on duty", "required on"],
        "description": "Choose how many workers start each shift so that every period's staffing requirement is met at minimum cost.",
        "formulation": (
            "Set: shifts S, periods T.  Data: coverage a_ts in {0,1}, requirement r_t, cost c_s.\n"
            "Vars: x[s] >= 0 integer.\n"
            "min sum_s c_s x_s   s.t.  sum_s a_ts x_s >= r_t  for all t"),
        "reasoning": ["define shift patterns and which periods each covers",
                      "one >= requirement constraint per period", "integer counts of people"],
        "errors": ["continuous workers", "coverage matrix built wrong for wrap-around days"],
    },
    {
        "name": "General LP/MILP",
        "keywords": ["minimize", "maximize", "constraint", "optimal", "optimize", "total"],
        "description": "Generic linear or mixed-integer program: decision quantities, a linear objective, linear constraints.",
        "formulation": (
            "Vars: x[i] with domains from the text (Binary / Integer / NonNegativeReals).\n"
            "min/max sum_i c_i x_i   s.t.  sum_i a_ji x_i (<=, =, >=) b_j"),
        "reasoning": ["list every decision quantity", "write the objective as a linear expression",
                      "translate every sentence with a limit into one linear constraint",
                      "check the direction of each inequality against the text"],
        "errors": ["products of decision variables", "missing the integrality stated in the text"],
    },
]
GENERIC_TEMPLATE = TEMPLATES[-1]


def template_score(t: dict, problem: str) -> float:
    """Eq. 24."""
    P = problem.lower()
    Q = words(problem)
    T = words(" ".join([t["name"], t["description"], t["formulation"],
                        " ".join(t["reasoning"]), " ".join(t["errors"]), " ".join(t["keywords"])]))
    s = 0.0
    for k in t["keywords"]:
        kw = k.lower()
        if kw in P:
            s += 5 + len(kw)
        s += 1.25 * len(words(kw) & Q)
    s += 0.15 * len(Q & T)
    return s


def retrieve_templates(problem: str, max_templates: int = C.MAX_TEMPLATES) -> list[dict]:
    """Templates with score >= 0.5 * max and >= 5, up to `max_templates`;
    otherwise the generic template."""
    scored = [(template_score(t, problem), t) for t in TEMPLATES[:-1]]
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0] if scored else 0.0
    keep = [t for s, t in scored if s >= 0.5 * best and s >= 5][:max_templates]
    return keep or [GENERIC_TEMPLATE]


def render_template(t: dict) -> str:
    return (f"### {t['name']}\n{t['description']}\n"
            f"Canonical formulation:\n{t['formulation']}\n"
            f"Reasoning flow: " + "; ".join(t["reasoning"]) + "\n"
            f"Common modelling errors: " + "; ".join(t["errors"]) + "\n")

"""Fixed constants of the SSC selector, transcribed from the paper
(Zeng et al., "MILP-Solve-Free Selection of LLM-Generated MILP Programs via
Multi-Signal Structural-Semantic Consistency").  Equation / table numbers refer
to that paper.  Nothing in this file is tuned; every number is the published one.
"""

# ---------------------------------------------------------------- Eq. 8 / 9
SYMBOLIC_WEIGHTS = {"SC": 0.35, "SQ": 0.20, "TA": 0.45}          # Eq. 8
FUSION_WEIGHTS = {"B": 0.65, "LP_SOL": 0.20, "LP_OBJ": 0.15}       # Eq. 9

# ---------------------------------------------------------------- Appendix H
CLONE_THRESHOLD = 0.95

# ---------------------------------------------------------------- Eq. 22
SIM_WEIGHTS = {"var": 0.18, "obj": 0.27, "con": 0.45, "dep": 0.05, "shape": 0.05}

# ---------------------------------------------------------------- Eq. 17
VAR_SIM_WEIGHTS = {"domain": 0.35, "arity": 0.30, "bounds": 0.15, "tokens": 0.20}
VAR_UNKNOWN_DOMAIN_MATCH = 0.4       # unknown vs known domain

# ---------------------------------------------------------------- Eq. 18
CON_SIM_WEIGHTS = {"n": 0.20, "s": 0.15, "a": 0.10, "t": 0.35, "v": 0.15, "p": 0.05}

# ---------------------------------------------------------------- Eq. 19
OBJ_SIM_WEIGHTS = {"direction": 0.35, "tokens": 0.25, "profile": 0.40}
OBJ_DIRECTION_UNKNOWN = 0.2
OBJ_PROFILE_WEIGHTS = {
    "numbers": 0.08, "operators": 0.10, "var_shapes": 0.18,
    "parameters": 0.17, "linear_terms": 0.37, "signed_numbers": 0.10,
}

# ---------------------------------------------------------------- Eq. 20
DEP_SIM_WEIGHTS = {"params": 0.75, "numbers": 0.25}

# ---------------------------------------------------------------- Eq. 21
SHAPE_SIM_WEIGHTS = {"n_vars": 0.25, "n_cons": 0.25, "has_obj": 0.20,
                     "domains": 0.15, "senses": 0.15}

# ---------------------------------------------------------------- Appendix A
LP_PROFILE_DIM = 30
LP_TIME_LIMIT_S = 5           # GLPK internal limit
LP_WATCHDOG_S = 20            # process-level watchdog
LP_RANGE_EPS = 1e-9

# ---------------------------------------------------------------- Section 2
EPS_REL = 1e-4
EPS_ABS = 1e-4
SCIP_TIME_LIMIT_S = 120
SANDBOX_TIMEOUT_S = 30

# ---------------------------------------------------------------- Table 8
# (penalty, fatal)
DEFECTS = {
    # Parse / AST
    "syntax_error": (1.00, True),
    "missing_variables": (0.30, True),
    "missing_objective": (0.30, True),
    "missing_constraints": (0.25, False),
    "unknown_objective_direction": (0.08, False),
    "unknown_variable_domain": (0.08, False),
    "unknown_constraint_sense": (0.06, False),
    "possible_nonlinearity": (0.15, False),
    # Pyomo API
    "reserved_pyomo_component_name": (0.60, True),
    "rule_arity_mismatch": (0.50, True),
    "unsupported_pyomo_max": (0.45, True),
    "unsupported_python_max_on_pyomo_expr": (0.10, False),
    "disallowed_component_decorator": (0.00, True),
    "disallowed_constraint_list": (0.00, True),
    "disallowed_add_call": (0.00, True),
    "disallowed_dynamic_component": (0.00, True),
    "disallowed_lambda_rule": (0.00, False),
    "contains_solver_call": (0.15, True),
    # Sandbox
    "sandbox_import_error": (0.70, True),
    "sandbox_construct_error": (0.70, True),
    "sandbox_timeout": (0.70, True),
    "sandbox_no_model": (0.65, True),
    "sandbox_no_variables": (0.45, True),
    "sandbox_no_objective": (0.45, True),
    "sandbox_no_constraints": (0.35, True),
    "sandbox_subprocess_error": (0.60, True),
    "sandbox_bad_json": (0.60, True),
    # Standardization
    "standardization_import_error": (0.70, True),
    "standardization_construct_error": (0.70, True),
    "standardization_no_model": (0.65, True),
    "standardization_no_variables": (0.45, True),
    "standardization_no_objective": (0.45, True),
    "standardization_no_constraints": (0.35, True),
    "standardization_timeout": (0.70, True),
    "standardization_subprocess_error": (0.70, True),
    "standardization_bad_json": (0.60, True),
}

# ---------------------------------------------------------------- Appendix L
TA_VOCABULARY = [
    "objective:minimize", "objective:maximize",
    "var:binary", "var:integer", "var:bounded",
    "constraint:equality", "constraint:equality_binary", "constraint:equality_continuous",
    "constraint:upper_bound", "constraint:lower_bound",
    "constraint:assignment", "constraint:capacity", "constraint:budget",
    "constraint:resource", "constraint:demand", "constraint:coverage",
    "constraint:linking", "constraint:facility_linking",
    "constraint:flow_balance", "constraint:inventory_balance", "constraint:routing",
]
TA_WEIGHT_MIN, TA_WEIGHT_MAX = 0.1, 2.0
TA_MAX_TOKENS = 1400
TA_RETRIES = 2

# ---------------------------------------------------------------- Appendix M
K = 10
TEMPERATURES = [0.6, 0.8, 1.0, 1.2, 1.4]
REPAIR_ROUNDS = 2
REPAIR_TEMPERATURE = 0.0
GENERATION_SEED = 0
MAX_TEMPLATES = 3
USE_TEMPLATES = True      # generation-side control switch (scripts/run.py --no-templates)

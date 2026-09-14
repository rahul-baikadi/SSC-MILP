"""Static (AST-level) checks of a candidate Pyomo program against the code
contract of Appendix M.6 and the Parse/AST + Pyomo-API rows of Table 8.

This is a reconstruction: the paper lists the defect names and penalties but
not the detection rules.  Each rule below is the most literal reading of the
defect name.  Returns a sorted list of defect names (duplicates removed) plus a
few structural facts reused by the semantic layer.
"""
from __future__ import annotations

import ast
from typing import Iterable

KNOWN_DOMAINS = {
    "Binary", "Boolean", "Integers", "NonNegativeIntegers", "PositiveIntegers",
    "NegativeIntegers", "NonPositiveIntegers", "Reals", "NonNegativeReals",
    "PositiveReals", "NegativeReals", "NonPositiveReals", "UnitInterval",
    "PercentFraction", "Any", "RangeSet",
}
# Attribute names of pyomo.environ.Block / ConcreteModel that a generated
# component must not shadow.  (Assigning e.g. model.name = pyo.Var() raises
# or silently breaks the model.)
RESERVED_COMPONENT_NAMES = {
    "name", "model", "index", "type", "display", "clone", "component",
    "components", "block", "parent", "dim", "keys", "values", "items",
    "data", "doc", "local_name", "active", "ctype", "construct",
    "add_component", "del_component", "solutions", "solve", "pprint",
    "write", "read", "load", "is_constructed", "component_map",
    "component_objects", "component_data_objects", "find_component",
    "deactivate", "activate", "reclassify_component_type",
    "collect_ctypes", "valid_model_component", "obj", "objective",
}
# "obj" is the contract's required objective name, so remove it again.
RESERVED_COMPONENT_NAMES -= {"obj", "objective"}

SOLVER_CALL_NAMES = {"SolverFactory", "value", "solve"}


def _attr_chain(node: ast.AST) -> str:
    """'pyo.Var' -> 'pyo.Var';  'model.x' -> 'model.x'; else ''."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _last(chain: str) -> str:
    return chain.rsplit(".", 1)[-1] if chain else ""


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.defects: set[str] = set()
        self.n_var = self.n_obj = self.n_con = 0
        self.model_names: set[str] = set()
        self.rule_funcs: dict[str, int] = {}      # func name -> #params
        self.rule_refs: list[tuple[str, int]] = []  # (func name, index dims)
        self.has_build_model = False

    # ------------------------------------------------------------ helpers
    def _is_pyomo_var_like_name(self, node: ast.AST) -> bool:
        """Heuristic: an attribute on the model object (model.x) or a
        subscript thereof (model.x[i]) is a Pyomo component reference."""
        while isinstance(node, ast.Subscript):
            node = node.value
        if isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            root = chain.split(".", 1)[0]
            return root in self.model_names or root in {"model", "m", "mdl"}
        return False

    # ------------------------------------------------------------ visits
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "build_model":
            self.has_build_model = True
        for d in node.decorator_list:
            chain = _attr_chain(d.func if isinstance(d, ast.Call) else d)
            if _last(chain) in {"Constraint", "Objective", "Expression"} and \
                    chain.split(".", 1)[0] in (self.model_names | {"model", "m"}):
                self.defects.add("disallowed_component_decorator")
        self.rule_funcs[node.name] = len(node.args.args) + len(node.args.posonlyargs)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        value = node.value
        if isinstance(value, ast.Call):
            chain = _attr_chain(value.func)
            if _last(chain) == "ConcreteModel":
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.model_names.add(t.id)
            if _last(chain) in {"AbstractModel"}:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.model_names.add(t.id)
        for t in node.targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                    and t.value.id in (self.model_names | {"model"}):
                if t.attr in RESERVED_COMPONENT_NAMES:
                    self.defects.add("reserved_pyomo_component_name")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        chain = _attr_chain(node.func)
        last = _last(chain)
        root = chain.split(".", 1)[0] if chain else ""

        # --- component constructors
        if last == "Var":
            self.n_var += 1
            dom = _kw(node, "within") or _kw(node, "domain")
            if dom is not None:
                dchain = _attr_chain(dom.func if isinstance(dom, ast.Call) else dom)
                if _last(dchain) not in KNOWN_DOMAINS:
                    self.defects.add("unknown_variable_domain")
        elif last == "Objective":
            self.n_obj += 1
            sense = _kw(node, "sense")
            if sense is None:
                self.defects.add("unknown_objective_direction")
            else:
                s = _attr_chain(sense)
                if _last(s) not in {"minimize", "maximize"}:
                    self.defects.add("unknown_objective_direction")
            self._check_rule(node, expected_dims=0)
        elif last == "Constraint":
            self.n_con += 1
            self._check_rule(node, expected_dims=len(node.args))
        elif last == "ConstraintList":
            self.defects.add("disallowed_constraint_list")
        elif last == "add" and isinstance(node.func, ast.Attribute) \
                and self._is_pyomo_var_like_name(node.func.value):
            self.defects.add("disallowed_add_call")
        elif last in {"add_component", "setattr"} and (root in self.model_names
                                                        or last == "setattr"):
            if last == "setattr" and node.args and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id in (self.model_names | {"model"}):
                self.defects.add("disallowed_dynamic_component")
            elif last == "add_component":
                self.defects.add("disallowed_dynamic_component")
        elif last in {"SolverFactory", "solve"} or (last == "value" and root in {"pyo", "pyomo", "pe", "value"}):
            self.defects.add("contains_solver_call")
        elif last == "max" and root in {"pyo", "pyomo", "pe"}:
            self.defects.add("unsupported_pyomo_max")
        elif last in {"max", "min"} and isinstance(node.func, ast.Name):
            if any(self._is_pyomo_var_like_name(a) for a in node.args):
                self.defects.add("unsupported_python_max_on_pyomo_expr")
        self.generic_visit(node)

    def _check_rule(self, node: ast.Call, expected_dims: int):
        rule = _kw(node, "rule") or _kw(node, "expr")
        if rule is None and node.args and isinstance(node.func, ast.Attribute) and node.func.attr == "Objective":
            return
        if isinstance(rule, ast.Lambda):
            self.defects.add("disallowed_lambda_rule")
        elif isinstance(rule, ast.Name):
            self.rule_refs.append((rule.id, expected_dims))

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, (ast.Mult, ast.Div, ast.Pow)):
            l_var = self._is_pyomo_var_like_name(node.left)
            r_var = self._is_pyomo_var_like_name(node.right)
            if isinstance(node.op, ast.Mult) and l_var and r_var:
                self.defects.add("possible_nonlinearity")
            if isinstance(node.op, ast.Div) and r_var:
                self.defects.add("possible_nonlinearity")
            if isinstance(node.op, ast.Pow) and l_var:
                self.defects.add("possible_nonlinearity")
        self.generic_visit(node)

    def finalize(self):
        for fname, dims in self.rule_refs:
            if fname in self.rule_funcs:
                n = self.rule_funcs[fname]
                # rule(model, *idx): 1 + dims params.  A rule for an indexed
                # component over a multi-dim set may legitimately take
                # 1+dims args where dims counts index sets; a 2-D RangeSet
                # product passed as one set is indistinguishable statically,
                # so only flag the unambiguous cases.
                if n == 0 or (dims == 0 and n > 1) or (dims > 0 and n < 2):
                    self.defects.add("rule_arity_mismatch")
        if self.n_var == 0:
            self.defects.add("missing_variables")
        if self.n_obj == 0:
            self.defects.add("missing_objective")
        if self.n_con == 0:
            self.defects.add("missing_constraints")


def static_check(code: str) -> dict:
    """Return {'defects': [...], 'has_build_model': bool, 'counts': {...}}."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"defects": ["syntax_error"], "has_build_model": False,
                "counts": {"var": 0, "obj": 0, "con": 0}}
    v = _Visitor()
    v.visit(tree)
    v.finalize()
    if not v.has_build_model:
        # Missing entry point is a construction failure at sandbox time; the
        # sandbox reports it, no separate static defect exists in Table 8.
        pass
    return {
        "defects": sorted(v.defects),
        "has_build_model": v.has_build_model,
        "counts": {"var": v.n_var, "obj": v.n_obj, "con": v.n_con},
    }


def repair_triggers(defects: Iterable[str]) -> list[str]:
    """Defects that trigger a repair round (Appendix M.4)."""
    trig = set()
    for d in defects:
        if d == "syntax_error" or d.startswith("reserved_") or d == "rule_arity_mismatch" \
                or d.startswith("disallowed_") or d.startswith("sandbox_") \
                or d.startswith("standardization_") or d.startswith("missing_") \
                or d == "possible_nonlinearity" or d == "contains_solver_call" \
                or d.startswith("unsupported_"):
            trig.add(d)
    return sorted(trig)

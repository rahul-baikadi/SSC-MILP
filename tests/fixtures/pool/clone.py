import pyomo.environ as pyo

DATA = {"boat_cap": 10, "canoe_cap": 8, "boat_time": 20, "canoe_time": 40,
        "max_boat_trips": 12, "canoe_frac": 0.6, "min_need": 300}

def build_model(data=None):
    d = data or DATA
    model = pyo.ConcreteModel()
    model.b = pyo.Var(within=pyo.NonNegativeIntegers)
    model.c = pyo.Var(within=pyo.NonNegativeIntegers)
    model.obj = pyo.Objective(expr=d["boat_time"] * model.b + d["canoe_time"] * model.c, sense=pyo.minimize)
    def need_rule(m):
        return d["boat_cap"] * m.b + d["canoe_cap"] * m.c >= d["min_need"]
    model.need = pyo.Constraint(rule=need_rule)
    def max_b_rule(m):
        return m.b <= d["max_boat_trips"]
    model.max_b = pyo.Constraint(rule=max_b_rule)
    def canoe_frac_rule(m):
        return m.c >= d["canoe_frac"] * (m.b + m.c)
    model.canoe_frac = pyo.Constraint(rule=canoe_frac_rule)
    return model
GOLD_ANSWER = None

import pyomo.environ as pyo

DATA = {"boat_cap": 100, "canoe_cap": 8, "boat_time": 20, "canoe_time": 40,
        "max_boat_trips": 12, "canoe_frac": 0.6, "min_ducks": 300}

def build_model(data=None):
    d = data or DATA
    model = pyo.ConcreteModel()
    model.boats = pyo.Var(within=pyo.NonNegativeIntegers)
    model.canoes = pyo.Var(within=pyo.NonNegativeIntegers)
    model.obj = pyo.Objective(expr=d["boat_time"] * model.boats + d["canoe_time"] * model.canoes, sense=pyo.minimize)
    def ducks_rule(m):
        return d["boat_cap"] * m.boats + d["canoe_cap"] * m.canoes >= d["min_ducks"]
    model.ducks = pyo.Constraint(rule=ducks_rule)
    def max_boats_rule(m):
        return m.boats <= d["max_boat_trips"]
    model.max_boats = pyo.Constraint(rule=max_boats_rule)
    def canoe_frac_rule(m):
        return m.canoes >= d["canoe_frac"] * (m.boats + m.canoes)
    model.canoe_frac = pyo.Constraint(rule=canoe_frac_rule)
    return model
GOLD_ANSWER = None

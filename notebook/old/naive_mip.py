# %%
from pulp import *

prob = LpProblem("myProblem", LpMinimize)
x = prob.add_variable("x", 0, 3)
# %%
y = prob.add_variable("y", cat="Binary")
prob += x + y <= 2
prob += -4 * x + y
stats = prob.solve(HiGHS())

# %%

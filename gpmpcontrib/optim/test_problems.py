import numpy as np
from gpmpcontrib.computerexperiment import ComputerExperiment

_TEST = False

def plog(x):
    return np.where(x >= 0, np.log(1 + x), -np.log(1 - x))

# ===== TestProblem01
_test_problem01_dict = {
    "input_dim": 2,
    "input_box": [[-1, -1], [1, 1]],
    "single_objective": lambda x: x[:, 0]**2 + x[:, 1]**2,
}

test_problem01 = ComputerExperiment(
    _test_problem01_dict["input_dim"],
    _test_problem01_dict["input_box"],
    single_objective=_test_problem01_dict["single_objective"]
)

if _TEST:
    print(test_problem01)
    test_problem01.eval(np.array([[0.0, 0.0]]))
    
# ===== TestProblem02
def _test_problem02_objective(x):
    return x[:, 0]**2 + x[:, 1]**2

def _test_problem02_constraint(x):
    return x[:, 0]**2 + x[:, 1]**2 - 2

_test_problem02_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "single_objective": _test_problem02_objective,
    "single_constraint": {
        "function": _test_problem02_constraint,
        "bounds": (0, 5)
    }
}

test_problem02 = ComputerExperiment(
    _test_problem02_dict["input_dim"],
    _test_problem02_dict["input_box"],
    single_objective=_test_problem02_dict["single_objective"],
    single_constraint=_test_problem02_dict["single_constraint"]
)

if _TEST:
    test_problem02.eval(np.array([[0.0, 0.0], [1.0, 1.0]]))

# ===== TestProblem03
_test_problem03_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "objective_list": [
        lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
        lambda x: -x[:, 0] ** 2 - x[:, 1] ** 2,
    ],
}

test_problem03 = ComputerExperiment(
    _test_problem03_dict["input_dim"],
    _test_problem03_dict["input_box"],
    objective_list=_test_problem03_dict["objective_list"]
)

if _TEST:
    test_problem03.eval(np.array([[0.0, 0.0], [1.0, 1.0]]))

# ===== TestProblem04
def _test_problem04_objective1(x):
    return x[:, 0]**2 + x[:, 1]**2

def _test_problem04_objective2(x):
    return -x[:, 0]**2 - x[:, 1]**2

_test_problem04_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "objective_list": [_test_problem04_objective1, _test_problem04_objective2],
    "single_constraint": {
        "function": _test_problem04_objective1,
        "bounds": (-np.inf, 5)
    }
}

test_problem04 = ComputerExperiment(
    _test_problem04_dict["input_dim"],
    _test_problem04_dict["input_box"],
    objective_list=_test_problem04_dict["objective_list"],
    single_constraint=_test_problem04_dict["single_constraint"]
)

if _TEST:
    test_problem04.eval(np.array([[0.0, 0.0], [1.0, 1.0]]))

# ===== G1
def _g1_objective(x):
    return 5 * (x[:, :3].sum(axis=1)) - 5 * (x[:, :4]**2).sum(axis=1) \
           - (x[:, 4:].sum(axis=1))

def _g1_constraints(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = x.T
    c1 = 2 * x1 + 2 * x2 + x10 + x11 - 10
    c2 = 2 * x1 + 2 * x3 + x10 + x12 - 10
    c3 = 2 * x2 + 2 * x3 + x11 + x12 - 10
    c4 = -8 * x1 + x10
    c5 = -8 * x2 + x11
    c6 = -8 * x3 + x12
    c7 = -2 * x4 - x5 + x10
    c8 = -2 * x6 - x7 + x11
    c9 = -2 * x8 - x9 + x12
    return np.column_stack((c1, c2, c3, c4, c5, c6, c7, c8, c9))

_g1_dict = {
    "input_dim": 13,
    "input_box": [[0] * 13, [1] * 9 + [100, 100, 100, 1]],
    "single_objective": _g1_objective,
    "single_constraint": {'function': _g1_constraints, 'output_dim': 9},
    "constraint_bounds": [[-np.inf, 0]] * 9
}

g1 = ComputerExperiment(
    _g1_dict["input_dim"],
    _g1_dict["input_box"],
    single_objective=_g1_dict["single_objective"],
    single_constraint=_g1_dict["single_constraint"],
    constraint_bounds=_g1_dict["constraint_bounds"]
)

if _TEST:
    print(g1)
    g1.eval(np.array([[0] * 13, [1] * 13]))
    g1.eval_constraints(np.array([[0] * 13, [1] * 13]))
    g1.eval_objectives(np.array([[0] * 13, [1] * 13]))

# ===== G3MOD_RR
def _g3mod_rr_objective(x):
    d = x.shape[1]
    return - plog(x.prod(axis=1) * np.sqrt(d)**d)

def _g3mod_rr_constraint(x):
    return (x**2).sum(axis=1) - 1

_g3mod_rr_dict = {
    "input_dim": 20,
    "input_box": [[0] * 20, [1] * 20],
    "single_objective": _g3mod_rr_objective,
    "single_constraint": _g3mod_rr_constraint,
    "constraint_bounds": [[-np.inf, 0]]
}

g3mod_rr = ComputerExperiment(
    _g3mod_rr_dict["input_dim"],
    _g3mod_rr_dict["input_box"],
    single_objective=_g3mod_rr_dict["single_objective"],
    single_constraint=_g3mod_rr_dict["single_constraint"],
    constraint_bounds=_g3mod_rr_dict["constraint_bounds"]
)

if _TEST:
    g3mod_rr.eval(np.array([[0] * 20]))

# ===== G5MOD_RR
def _g5mod_rr_objective(x):
    return 3 * x[:, 0] + 1e-6 * x[:, 0]**3 + 2 * x[:, 1] + (2 / 3) * 1e-6 * x[:, 1]**3

def _g5mod_rr_constraints(x):
    c1 = x[:, 2] - x[:, 3] - 0.55
    c2 = x[:, 3] - x[:, 2] - 0.55
    c3 = 1000 * (np.sin(-x[:, 2] - 0.25) + np.sin(-x[:, 3] - 0.25)) + 894.8 - x[:, 0]
    c4 = 1000 * (np.sin(x[:, 2] - 0.25) + np.sin(x[:, 2] - x[:, 3] - 0.25)) + 894.8 - x[:, 1]
    c5 = 1000 * (np.sin(x[:, 3] - 0.25) + np.sin(x[:, 3] - x[:, 2] - 0.25)) + 1294.8
    return np.column_stack((c1, c2, c3, c4, c5))

_g5mod_rr_dict = {
    "input_dim": 4,
    "input_box": [[0, 0, -0.55, -0.55], [1200, 1200, 0.55, 0.55]],
    "single_objective": _g5mod_rr_objective,
    "single_constraint": {"function": _g5mod_rr_constraints, "output_dim": 5, "bounds": [[-np.inf, 0.0]] * 5}
}

g5mod_rr = ComputerExperiment(
    _g5mod_rr_dict["input_dim"],
    _g5mod_rr_dict["input_box"],
    single_objective=_g5mod_rr_dict["single_objective"],
    single_constraint=_g5mod_rr_dict["single_constraint"]
)

if _TEST:
    g5mod_rr.eval(np.array([[0] * 4]))

# ===== G6
def _g6_objective(x):
    return (x[:, 0] - 10)**3 + (x[:, 1] - 20)**3

def _g6_constraints(x):
    c1 = - (x[:, 0] - 5)**2 - (x[:, 1] - 5)**2 + 100
    c2 = (x[:, 0] - 6)**2 + (x[:, 1] - 5)**2 - 82.81
    return np.column_stack((c1, c2))

_g6_dict = {
    "input_dim": 2,
    "input_box": [[13, 0], [100, 100]],
    "single_objective": _g6_objective,
    "single_constraint": {'function': _g6_constraints, 'output_dim': 2, 'bounds': [[100., np.inf], [-np.inf, 82.81]]}
}

g6 = ComputerExperiment(
    _g6_dict["input_dim"],
    _g6_dict["input_box"],
    single_objective=_g6_dict["single_objective"],
    single_constraint=_g6_dict["single_constraint"]
)

if _TEST:
    g6.eval(np.array([[50.0, 50.0], [80., 80.]]))

# ===== G7
def _g7_objective(x):
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1] - 14 * x[:, 0] - 16 * x[:, 1] + (x[:, 2] - 10)**2 + 4 * (x[:, 3] - 5)**2 + (x[:, 4] - 3)**2 + 2 * (x[:, 5] - 1)**2 + 5 * x[:, 6]**2 + 7 * (x[:, 7] - 11)**2 + 2 * (x[:, 8] - 10)**2 + (x[:, 9] - 7)**2 + 45

def _g7_constraints(x):
    c1 = (4 * x[:, 0] + 5 * x[:, 1] - 3 * x[:, 6] + 9 * x[:, 7] - 105) / 105
    c2 = (10 * x[:, 0] - 8 * x[:, 1] - 17 * x[:, 6] + 2 * x[:, 7]) / 370
    c3 = (-8 * x[:, 0] + 2 * x[:, 1] + 5 * x[:, 8] - 2 * x[:, 9] - 12) / 158
    c4 = (3 * (x[:, 0] - 2)**2 + 4 * (x[:, 1] - 3)**2 + 2 * x[:, 2]**2 - 7 * x[:, 3] - 120) / 1258
    c5 = (5 * x[:, 0]**2 + 8 * x[:, 1] + (x[:, 2] - 6)**2 - 2 * x[:, 3] - 40) / 816
    c6 = (0.5 * (x[:, 0] - 8)**2 + 2 * (x[:, 1] - 4)**2 + 3 * x[:, 4]**2 - x[:, 5] - 30) / 834
    c7 = (x[:, 0]**2 + 2 * (x[:, 1] - 2)**2 - 2 * x[:, 0] * x[:, 1] + 14 * x[:, 4] - 6 * x[:, 5]) / 788
    c8 = (-3 * x[:, 0] + 6 * x[:, 1] + 12 * (x[:, 8] - 8)**2 - 7 * x[:, 9]) / 4048
    return np.column_stack((c1, c2, c3, c4, c5, c6, c7, c8))

_g7_dict = {
    "input_dim": 10,
    "input_box": [[-10] * 10, [10] * 10],
    "single_objective": _g7_objective,
    "single_constraint": {"function": _g7_constraints, "output_dim": 8, "bounds": [[-np.inf, 0.0]]*8}
}

g7 = ComputerExperiment(
    _g7_dict["input_dim"],
    _g7_dict["input_box"],
    single_objective=_g7_dict["single_objective"],
    single_constraint=_g7_dict["single_constraint"]
)

if _TEST:
    g7.eval(np.array([[0.0]*10]))

# ===== G8
def _g8_objective(x):
    return - (np.sin(2 * np.pi * x[:, 0]) ** 3) * np.sin(2 * np.pi * x[:, 1]) / ((x[:, 0] + x[:, 1]) * x[:, 0] ** 3)

def _g8_constraints(x):
    c1 = x[:, 0] ** 2 - x[:, 1] + 1
    c2 = 1 - x[:, 0] + (x[:, 1] - 4) ** 2
    return np.column_stack((c1, c2))

_g8_dict = {
    "input_dim": 2,
    "input_box": [[0, 0], [10, 10]],
    "single_objective": _g8_objective,
    "single_constraint": {'function': _g8_constraints, 'output_dim': 2, "bounds": [[-np.inf, 0.0]]*2}
}

g8 = ComputerExperiment(
    _g8_dict["input_dim"],
    _g8_dict["input_box"],
    single_objective=_g8_dict["single_objective"],
    single_constraint=_g8_dict["single_constraint"]
)

if _TEST:
    g8.eval(np.array([[0.25] * 2]))

# ===== G9 =====
def _g9_objective(x):
    obj = (x[:, 0] - 10) ** 2 + 5 * (x[:, 1] - 12) ** 2 \
          + x[:, 2] ** 4 + 3 * (x[:, 3] - 11) ** 2 + 10 * x[:, 4] ** 6 \
          + 7 * x[:, 5] ** 2 + x[:, 6] ** 4 - 4 * x[:, 5] * x[:, 6] \
          - 10 * x[:, 5] - 8 * x[:, 6]
    return obj

def _g9_constraints(x):
    v1 = 2 * x[:, 0] ** 2
    v2 = x[:, 1] ** 2
    c1 = (v1 + 3 * v2 ** 2 + x[:, 2] + 4 * x[:, 3] ** 2 + 5 * x[:, 4] - 127.) / 127.
    c2 = (7 * x[:, 0] + 3 * x[:, 1] + 10 * x[:, 2] ** 2 + x[:, 3] - x[:, 4] - 282.) / 282.
    c3 = (23 * x[:, 0] + v2 + 6 * x[:, 5] ** 2 - 8 * x[:, 6] - 196.) / 196.
    c4 = 2 * v1 + v2 - 3 * x[:, 0] * x[:, 1] + 2 * x[:, 2] ** 2 + 5 * x[:, 5] - 11 * x[:, 6]
    return np.column_stack((c1, c2, c3, c4))

_g9_dict = {
    "input_dim": 7,
    "input_box": [[-10] * 7, [10] * 7],
    "single_objective": _g9_objective,
    "single_constraint": {'function': _g9_constraints, 'output_dim': 4, "bounds": [[-np.inf, 0.0]]*4}
}

g9 = ComputerExperiment(
    _g9_dict["input_dim"],
    _g9_dict["input_box"],
    single_objective=_g9_dict["single_objective"],
    single_constraint=_g9_dict["single_constraint"]
)

# ===== G10 =====

def _g10_objective(x):
    return x[:, 0] + x[:, 1] + x[:, 2]

def _g10_constraints(x):
    c1 = -1 + 0.0025 * (x[:, 3] + x[:, 5])
    c2 = -1 + 0.0025 * (-x[:, 3] + x[:, 4] + x[:, 6])
    c3 = -1 + 0.01 * (-x[:, 4] + x[:, 7])
    c4 = 100 * x[:, 0] - x[:, 0] * x[:, 5] + 833.33252 * x[:, 3] - 83333.333
    c5 = x[:, 1] * x[:, 3] - x[:, 1] * x[:, 6] - 1250 * x[:, 3] + 1250 * x[:, 4]
    c6 = x[:, 2] * x[:, 4] - x[:, 2] * x[:, 7] - 2500 * x[:, 4] + 1250000
    return np.column_stack((c1, c2, c3, c4, c5, c6))

_g10_dict = {
    "input_dim": 8,
    "input_box": [[100, 1000, 1000, 10, 10, 10, 10, 10], [10000] * 3 + [1000] * 5],
    "single_objective": _g10_objective,
    "single_constraint": {"function": _g10_constraints, "output_dim": 6, "bounds": [[-np.inf, 0]] * 6}
}

g10 = ComputerExperiment(
    _g10_dict["input_dim"],
    _g10_dict["input_box"],
    single_objective=_g10_dict["single_objective"],
    single_constraint=_g10_dict["single_constraint"]
)

# ===== G10MOD =====
def _g10mod_constraints(x):
    raw_constraints = _g10_constraints(x)
    raw_constraints[:, [3, 4, 5]] = np.log1p(raw_constraints[:, [3, 4, 5]])
    return raw_constraints

_g10mod_dict = _g10_dict.copy()
_g10mod_dict["single_constraint"] = {"function": _g10mod_constraints, "output_dim": 6, "bounds": [[-np.inf, 0]] * 6}

g10mod = ComputerExperiment(
    _g10mod_dict["input_dim"],
    _g10mod_dict["input_box"],
    single_objective=_g10mod_dict["single_objective"],
    single_constraint=_g10mod_dict["single_constraint"]
)

# ===== G10MODMOD =====
def _g10modmod_constraints(x):
    raw_constraints = _g10mod_constraints(x)
    raw_constraints[:, [3, 4, 5]] = raw_constraints[:, [3, 4, 5]] ** 7
    return raw_constraints

_g10modmod_dict = _g10_dict.copy()
_g10modmod_dict["single_constraint"] = {"function": _g10modmod_constraints, "output_dim": 6, "bounds": [[-np.inf, 0]] * 6}

g10modmod = ComputerExperiment(
    _g10modmod_dict["input_dim"],
    _g10modmod_dict["input_box"],
    single_objective=_g10modmod_dict["single_objective"],
    single_constraint=_g10modmod_dict["single_constraint"]
)

# ===== G13MOD =====
def _g13mod_objective(x):
    obj = np.prod(x, axis=1)
    obj = np.log(np.exp(obj))
    return obj.reshape(-1, 1)

def _g13mod_constraints(x):
    c1 = (x ** 2).sum(axis=1) - 10
    c2 = x[:, 1] * x[:, 2] - 5 * x[:, 3] * x[:, 4]
    c3 = x[:, 0] ** 3 + x[:, 1] ** 3 + 1
    return np.column_stack((c1, c2, c3))

_g13mod_dict = {
    "input_dim": 5,
    "input_box": [[-2.3, -2.3, -3.2, -3.2, -3.2], [2.3, 2.3, 3.2, 3.2, 3.2]],
    "single_objective": _g13mod_objective,
    "single_constraint": {"function": _g13mod_constraints, "output_dim": 3, "bounds": [[-np.inf, 0]] * 3}
}

g13mod = ComputerExperiment(
    _g13mod_dict["input_dim"],
    _g13mod_dict["input_box"],
    single_objective=_g13mod_dict["single_objective"],
    single_constraint=_g13mod_dict["single_constraint"]
)

# ====== G16
def _g16_evaluation(x):
    x1, x2, x3, x4, x5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    y1 = x2 + x3 + 41.6
    c1 = 0.024 * x4 - 4.62
    y2 = 12.5 / c1 + 12
    c2 = 0.0003535 * x1 ** 2 + 0.5311 * x1 + 0.08705 * y2 * x1
    c3 = 0.052 * x1 + 78 + 0.002377 * y2 * x1
    y3 = c2 / c3
    y4 = 19 * y3
    c4 = 0.04782 * (x1 - y3) + 0.1956 * (x1 - y3) ** 2. / x2 + 0.6376 * y4 + 1.594 * y3
    c5 = 100 * x2
    c6 = x1 - y3 - y4
    c7 = 0.950 - c4 / c5
    y5 = c6 * c7
    y6 = x1 - y5 - y4 - y3
    c8 = (y5 + y4) * 0.995
    y7 = c8 / y1
    y8 = c8 / 3798
    c9 = y7 - 0.0663 * y7 / y8 - 0.3153
    y9 = 96.82 / c9 + 0.321 * y1
    y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
    y11 = 1.71 * x1 - 0.452 * y4 + 0.580 * y3
    c10 = 12.3 / 752.3
    c11 = (1.75 * y2) * (0.995 * x1)
    c12 = 0.995 * y10 + 1998
    y12 = c10 * x1 + c11 / c12
    y13 = c12 + 1.75 * y2
    y14 = 3623 + 64.4 * x2 + 58.4 * x3 + 146312. / (y9 + x5)
    c13 = 0.995 * y10 + 60.8 * x2 + 48 * x4 - 0.1121 * y14 - 5095
    y15 = y13 / c13
    y16 = 148000 - 331000 * y15 + 40 * y13 - 61 * y15 * y13
    c14 = 2324 * y10 - 28740000 * y2
    y17 = 14130000 - 1328 * y10 - 531 * y11 + c14 / c12
    c15 = y13 / y15 - y13 / 0.52
    c16 = 1.104 - 0.72 * y15
    c17 = y9 + x5

    obj = 0.000117 * y14 + 0.1365 + 0.00002358 * y13 + 0.000001502 * y16 + 0.0321 * y12 + 0.004324 * y5 + 0.0001 * c15 / c16 + 37.48 * y2 / c12 - 0.0000005843 * y17
    obj = obj.reshape(-1, 1)

    c = np.zeros([x.shape[0], 38])
    # Fill the constraints array with values (c[:, 0] to c[:, 37])
    c[:, 0] = 0.28 / 0.72 * y5 - y4
    c[:, 1] = x3 - 1.5 * x2
    c[:, 2] = 3496 * y2 / c12 - 21
    c[:, 3] = 110.6 + y1 - 62212. / c17
    c[:, 4] = 213.1 - y1
    c[:, 5] = y1 - 405.23
    c[:, 6] = 17.505 - y2
    c[:, 7] = y2 - 1053.6667
    c[:, 8] = 11.275 - y3
    c[:, 9] = y3 - 35.03
    c[:, 10] = 214.228 - y4
    c[:, 11] = y4 - 665.585
    c[:, 12] = 7.458 - y5
    c[:, 13] = y5 - 584.463
    c[:, 14] = 0.961 - y6
    c[:, 15] = y6 - 265.916
    c[:, 16] = 1.612 - y7
    c[:, 17] = y7 - 7.046
    c[:, 18] = 0.146 - y8
    c[:, 19] = y8 - 0.222
    c[:, 20] = 107.99 - y9
    c[:, 21] = y9 - 273.366
    c[:, 22] = 922.693 - y10
    c[:, 23] = y10 - 1286.105
    c[:, 24] = 926.832 - y11
    c[:, 25] = y11 - 1444.046
    c[:, 26] = 18.766 - y12
    c[:, 27] = y12 - 537.141
    c[:, 28] = 1072.163 - y13
    c[:, 29] = y13 - 3247.039
    c[:, 30] = 8961.448 - y14
    c[:, 31] = y14 - 26844.086
    c[:, 32] = 0.063 - y15
    c[:, 33] = y15 - 0.386
    c[:, 34] = 71084.33 - y16
    c[:, 35] = -140000 + y16
    c[:, 36] = 2802713 - y17
    c[:, 37] = y17 - 12146108

    return np.column_stack((obj, c))

_g16_dict = {
    "input_dim": 5,
    "input_box": [[704.4148,  68.6   ,   0.    , 193.    ,  25.    ],
                  [906.3855, 288.88  , 134.75  , 287.0966,  84.1988]],
    "single_function": {'function': _g16_evaluation,
                        'output_dim': 1+38,
                        'type': ["objective"] + ["constraint"] * 38,
                        'bounds': [None] + [[-np.inf, 0.0]]*38}
}

g16 = ComputerExperiment(
    _g16_dict["input_dim"],
    _g16_dict["input_box"],
    single_function=_g16_dict["single_function"]
)
if _TEST:
    g16.eval(np.array([[704.4148,  68.6   ,   0.    , 193.    ,  25.    ]]))
    g16.eval_objectives(np.array([[704.4148,  68.6   ,   0.    , 193.    ,  25.    ]]))
    g16.eval_constraints(np.array([[704.4148,  68.6   ,   0.    , 193.    ,  25.    ]]))
# =====  G18 problem
def _g18_evaluation(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9 = x.T
    
    c1 = 1 - x3**2 - x4**2
    c2 = 1 - x9**2
    c3 = 1 - x5**2 - x6**2
    c4 = 1 - x1**2 - (x2-x9)**2
    c5 = 1 - (x1-x5)**2 - (x2-x6)**2
    c6 = 1 - (x1-x7)**2 - (x2-x8)**2
    c7 = 1 - (x3-x5)**2 - (x4-x6)**2
    c8 = 1 - (x3-x7)**2 - (x4-x8)**2
    c9 = 1 - x7**2 - (x8-x9)**2
    c10 = x1*x4 - x2*x3
    c11 = x3*x9
    c12 = -x5*x9
    c13 = x5*x8 - x6*x7
    
    obj = 0.5 * (x1*x4 - x2*x3 + x3*x9 - x5*x9 + x5*x8 - x6*x7)

    constraints = np.column_stack((-c1, -c2, -c3, -c4, -c5, -c6, -c7, -c8, -c9, -c10, -c11, -c12, -c13))

    return np.column_stack((obj, constraints))

_g18_dict = {
    "input_dim": 9,
    "input_box": [[-10]*9, [10]*9],
    "single_function": {'function': _g18_evaluation,
                        'output_dim': 14,
                        'type': ['objective'] + ['constraints']*13,
                        'bounds': [None] + [[-np.inf, 0]] * 13}
}

g18 = ComputerExperiment(
    _g18_dict["input_dim"],
    _g18_dict["input_box"],
    single_function=_g18_dict["single_function"]
)

# ===== G19

def _g19_evaluation(x):
    n = x.shape[0]

    b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
    e = [-15, - 27, - 36, - 18, - 12]

    c = np.array([
        [30, -20, -10, 32, -10],
        [-20, 39, -6, -31, 32],
        [-10, -6, 10, -6, -10],
        [32, -31, -6, 39, -20],
        [-10, 32, -10, -20, 30]
    ])

    d = [4, 8, 10, 6, 2]

    a = np.array([
        [-16, 2, 0, 1, 0],
        [0, -2, 0, 0.4, 2],
        [-3.5, 0, 2, 0, 0],
        [0, -2, 0, -4, -1],
        [0, -9, -2, 1, -2.8],
        [2, 0, -4, 0, 0],
        [-1, -1, -1, -1, -1],
        [-1, -2, -3, -2, -1],
        [1, 2, 3, 4, 5],
        [1, 1, 1, 1, 1]
    ])

    sum1 = np.zeros(n)
    for j in range(5):
        for i in range(5):
            sum1 = sum1 + c[i, j] * x[:, 10 + i] * x[:, 10 + j]

    sum2 = np.zeros(n)
    for j in range(5):
        sum2 = sum2 + d[j] * (x[:, 10+j] ** 3)

    sum3 = np.zeros(n)
    for i in range(10):
        sum3 = sum3 + b[i] * x[:, i]

    obj = sum1 + 2 * sum2 - sum3
    obj = obj.reshape(-1, 1)

    constraints = np.zeros([n, 5])
    for j in range(5):

        sum1 = np.zeros(n)
        for i in range(5):
            sum1 = sum1 + c[i, j] * x[:, 10 + i]

        sum2 = np.zeros(n)
        for i in range(10):
            sum2 = sum2 + a[i, j] * x[:, i]

        constraints[:, j] = -2 * sum1 - e[j] + sum2

    return np.column_stack((obj, constraints))

_g19_dict = {
    "input_dim": 15,
    "input_box": [[0]*15, [10]*15],
    "single_function": {'function': _g19_evaluation,
                        'output_dim': 1+5,
                        'type': ["objective"] + ["constraint"] * 5,
                        'bounds': [None] + [[-np.inf, 0]] * 5}
}

g19 = ComputerExperiment(
    _g19_dict["input_dim"],
    _g19_dict["input_box"],
    single_function=_g19_dict["single_function"]
)

# ===== G24 ======
def _g24_evaluation(x):
    x1 = x[:, 0]
    x2 = x[:, 1]

    obj = - x1 - x2
    
    c1 = - 2 * x1 ** 4 + 8 * x1 ** 3 - 8 * x1 ** 2 + x2 - 2
    c2 = - 4 * x1 ** 4 + 32 * x1 ** 3 - 88 * x1 ** 2 + 96 * x1 + x2 - 36

    constraints = np.column_stack((c1, c2))
    
    return np.column_stack((obj, constraints))

_g24_dict = {
    "input_dim": 2,
    "input_box": [[0, 0], [3, 4]],
    "single_function": {'function': _g24_evaluation,
                        'output_dim': 1+2,
                        'type': ["objective"] + ["constraint"] * 2,
                        'bounds': [None] + [[-np.inf, 0]] * 2}
}

g24 = ComputerExperiment(
    _g24_dict["input_dim"],
    _g24_dict["input_box"],
    single_function=_g24_dict["single_function"]
)

# ==== GoldsteinPrice function ====
#
# See https://www.sfu.ca/~ssurjano/goldpr.html
#
# References:
# Dixon, L. C. W., & Szego, G. P. (1978). The global optimization problem: an introduction. Towards global optimization, 2, 1-15.
# Molga, M., & Smutnicki, C. Test functions for optimization needs (2005). Retrieved June 2013, from http://www.zsd.ict.pwr.wroc.pl/files/docs/functions.pdf.
#

def _goldsteinprice_objective(x):
    obj = (1 + ((x[:, 0] + x[:, 1] + 1)**2) * (19 - 14 * x[:,0] + 3 * x[:, 0]**2 - 14 * x[:,1] +  6 * x[:, 0] * x[:, 1] + 3 * x[:,1]**2))
    obj = obj * (30 + ((2 * x[:,0] - 3 * x[:, 1])**2) * (18 - 32 * x[:, 0] + 12 * x[:,0]**2 + 48 * x[:,1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:,1]**2))
    return obj

_goldsteinprice_dict = {
    "input_dim": 2,
    "input_box": [[-2, -2], [2, 2]],
    "single_objective": _goldsteinprice_objective,
}

goldsteinprice = ComputerExperiment(
    _goldsteinprice_dict["input_dim"],
    _goldsteinprice_dict["input_box"],
    single_objective=_goldsteinprice_dict["single_objective"],
)

#  ==== log-GoldsteinPrice function ====

def _goldstein_price_log_objective(x):
    # Placeholder function for GoldsteinPrice
    obj_gp = _goldsteinprice_objective(x).reshape(-1, 1)
    return np.log(obj_gp)

_goldstein_price_log_dict = {
    "input_dim": 2,
    "input_box": [[-2, -2], [2, 2]],
    "single_objective": _goldstein_price_log_objective,
}

goldstein_price_log = ComputerExperiment(
    _goldstein_price_log_dict["input_dim"],
    _goldstein_price_log_dict["input_box"],
    single_objective=_goldstein_price_log_dict["single_objective"],
)

# ===== Shekel Problems ======
# See https://www.sfu.ca/~ssurjano/shekel.html
#
def create_shekel_problem(m):
    C = np.array([
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
        [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
        [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
    ])

    beta = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])/10

    def _shekel_objective(x):
        sum1 = 0
        for i in range(m):
            sum2 = 0
            for j in range(x.shape[1]):
                sum2 = sum2 + (x[:, j] - C[j, i]) ** 2
            sum1 = sum1 - 1/(sum2 + beta[i])
        return sum1

    return {
        "input_dim": 4,
        "input_box": [[0]*4, [10]*4],
        "single_objective": _shekel_objective,
    }

# Create the shekel problems
_shekel10_dict = create_shekel_problem(10)
shekel10 = ComputerExperiment(
    _shekel10_dict["input_dim"],
    _shekel10_dict["input_box"],
    single_objective=_shekel10_dict["single_objective"],
)

_shekel7_dict = create_shekel_problem(7)
shekel7 = ComputerExperiment(
    _shekel7_dict["input_dim"],
    _shekel7_dict["input_box"],
    single_objective=_shekel7_dict["single_objective"],
)

_shekel5_dict = create_shekel_problem(5)
shekel5 = ComputerExperiment(
    _shekel5_dict["input_dim"],
    _shekel5_dict["input_box"],
    single_objective=_shekel5_dict["single_objective"],
)

# ===== Harmann problems =====
# See https://www.sfu.ca/~ssurjano/hart3.html
# and https://www.sfu.ca/~ssurjano/hart6.html
def _hartman3_objective(x):
    alpha = np.array([1, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10, 30],
        [0.1, 10, 35],
        [3.0, 10, 30],
        [0.1, 10, 35]
    ])
    P = (10 ** (-4)) * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828]
    ])

    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(x.shape[1]):
            sum2 = sum2 - A[i, j] * (x[:, j] - P[i, j]) ** 2

        sum1 = sum1 - alpha[i] * np.exp(sum2)

    return sum1


_hartman3_dict = {
    "input_dim": 3,
    "input_box": [[0] * 3, [1] * 3],
    "single_objective": _hartman3_objective
}

hartman3 = ComputerExperiment(
    _hartman3_dict["input_dim"],
    _hartman3_dict["input_box"],
    single_objective=_hartman3_dict["single_objective"]
)



def _hartman6_objective(X):
    alpha = np.array([1, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = (10 ** (-4)) * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])

    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(X.shape[1]):
            sum2 = sum2 - A[i, j] * (X[:, j] - P[i, j]) ** 2

        sum1 = sum1 - alpha[i] * np.exp(sum2)

    return sum1


_hartman6_dict = {
    "input_dim": 6,
    "input_box": [[0] * 6, [1] * 6],
    "single_objective": _hartman6_objective
}

hartman6 = ComputerExperiment(
    _hartman6_dict["input_dim"],
    _hartman6_dict["input_box"],
    single_objective=_hartman6_dict["single_objective"]
)


# ===== PVD4 ======
# See 
def _pvd4_objective(x):
    x1, x2, x3, x4 = x.T

    obj = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2 + 3.1661 * (x1 ** 2) * x4 + 19.84 * (x1 ** 2) * x3
    
    return obj

def _pvd4_constraints(x):
    x1, x2, x3, x4 = x.T
    
    c1 = - x1 + 0.0193 * x3
    c2 = - x2 + 0.00954 * x3
    c3 = -np.pi * (x3 ** 2) * x4 - (4/3) * np.pi * (x3 ** 3) + 1296000
    
    return np.column_stack((c1, c2, c3))

_pvd4_dict = {
    "input_dim": 4,
    "input_box": [[0] * 4, [1, 1, 50, 240]],
    "single_objective": _pvd4_objective,
    "single_constraint": {'function': _pvd4_constraints, 'output_dim': 3},
    "constraint_bounds": [[-np.inf, 0]] * 3
}

pvd4 = ComputerExperiment(
    _pvd4_dict["input_dim"],
    _pvd4_dict["input_box"],
    single_objective=_pvd4_dict["single_objective"],
    single_constraint=_pvd4_dict["single_constraint"],
    constraint_bounds=_pvd4_dict["constraint_bounds"]
)


# ===== SR7 =====
def _sr7_objectives(x):
    x1, x2, x3, x4, x5, x6, x7 = x.T

    A = (3.3333 * (x3 ** 2)) + (14.9334 * x3) - 43.0934
    B = (x6 ** 2) + (x7 ** 2)
    C = (x6 ** 3) + (x7 ** 3)
    D = (x4 * (x6 ** 2)) + (x5 * (x7 ** 2))

    obj = (0.7854 * x1 * (x2 ** 2) * A) - (1.508 * x1 * B) + (7.477 * C) + (0.7854 * D)
    return obj

def _sr7_constraints(x):
    x1, x2, x3, x4, x5, x6, x7 = x.T

    A1 = np.sqrt((745 * x4 / (x2 * x3))**2 + 16.91 * (10 ** 6))
    A2 = np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5 * (10 ** 6))

    B1 = 0.1 * (x6 ** 3)
    B2 = 0.1 * (x7 ** 3)

    c = np.zeros([x.shape[0], 11])
    c[:, 0] = (27 - (x1 * (x2 ** 2) * x3))/27
    c[:, 1] = (397.5 - (x1 * (x2 ** 2) * (x3 ** 2)))/397.5
    c[:, 2] = (1.93 - (x2 * (x6 ** 4) * x3) / (x4 ** 3)) / 1.93
    c[:, 3] = (1.93 - (x2 * (x7 ** 4) * x3) / (x5 ** 3)) / 1.93
    c[:, 4] = ((A1/B1) - 1100)/1100
    c[:, 5] = ((A2/B2) - 850)/850
    c[:, 6] = ((x2 * x3) - 40)/40
    c[:, 7] = (5 - (x1/x2))/5
    c[:, 8] = ((x1/x2) - 12)/12
    c[:, 9] = (1.9 + 1.5 * x6 - x4)/1.9
    c[:, 10] = (1.9 + 1.1 * x7 - x5)/1.9
    return c

_sr7_dict = {
    "input_dim": 7,
    "input_box": [[2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0],
                  [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]],
    "single_objective": _sr7_objectives,
    "single_constraint": {'function': _sr7_constraints, 'output_dim': 11},
    "constraint_bounds": [[-np.inf, 0]] * 11
}

sr7 = ComputerExperiment(
    _sr7_dict["input_dim"],
    _sr7_dict["input_box"],
    single_objective=_sr7_dict["single_objective"],
    single_constraint=_sr7_dict["single_constraint"],
    constraint_bounds=_sr7_dict["constraint_bounds"]
)

# ===== BNH ======

def _bnh_objectives(x):
    x1, x2 = x.T

    f1 = 4*x1**2 + 4*x2**2
    f2 = (x1-5)**2 + (x2-5)**2
    return np.column_stack((f1, f2))

def _bnh_constraints(x):
    x1, x2 = x.T

    c1 = (x1-5)**2 + x2**2 - 25
    c2 = 7.7 - (x1-8)**2 - (x2+3)**2
    return np.column_stack((c1, c2))

_bnh_dict = {
    "input_dim": 2,
    "input_box": [[0, 0], [5, 3]],
    "objective_list": [
        {"function": lambda x: _bnh_objectives(x)[:, 0], "name": "f1", "goal": "minimize"},
        {"function": lambda x: _bnh_objectives(x)[:, 1], "name": "f2", "goal": "minimize"}
    ],
    "constraint_list": [
        {"function": lambda x: _bnh_constraints(x)[:, 0], "name": "c1", "bounds": [None, 0]},
        {"function": lambda x: _bnh_constraints(x)[:, 1], "name": "c2", "bounds": [None, 0]}
    ]
}

bnh = ComputerExperiment(
    _bnh_dict["input_dim"],
    _bnh_dict["input_box"],
    objective_list=_bnh_dict["objective_list"],
    constraint_list=_bnh_dict["constraint_list"],
)

# ===== WeldedBeamDesign =====

def _welded_beam_design_objectives(x):
    x1, x2, x3, x4 = x.T
    f1 = 1.10471*x2*x1**2 + 0.04811*x3*x4*(14 + x2)
    f2 = 2.1952/(x4*x3**3)
    return np.column_stack((f1, f2))

def _welded_beam_design_constraints(x):
    x1, x2, x3, x4 = x.T

    tau_max = 13600
    sigma_max = 30000
    pc_min = 6000

    tau_prime = 6000/(np.sqrt(2) * x1 * x2)
    tau_second = (6000*(14 + 0.5*x2)*np.sqrt(0.25*(x2**2 + (x1 + x3)**2)))/(2*np.sqrt(2) * x1 * x2 * ((x2**2)/12 + 0.25 * (x1 + x3)**2))
    tau = np.sqrt(tau_prime**2 + tau_second ** 2 + (x2 * tau_prime * tau_second)/np.sqrt(0.25*(x2**2 + (x1+x3)**2)))

    sigma = 504000 / (x4 * x3 ** 2)

    pc = 64746.022 * (1 - 0.0282346 * x3) * x3 * x4 ** 3

    c1 = -(tau_max - tau)
    c2 = -(sigma_max - sigma)
    c3 = -(x4 - x1)
    c4 = -(pc - pc_min)

    return np.column_stack((c1, c2, c3, c4))

_welded_beam_design_dict = {
    "input_dim": 4,
    "input_box": [[0.125, 0.1, 0.1, 0.125], [5, 10, 10, 5]],
    "single_objective": {'function': _welded_beam_design_objectives, 'output_dim': 2},
    "single_constraint": {'function': _welded_beam_design_constraints, 'output_dim': 4},
    "constraint_bounds": [[-np.inf, 0]] * 4
}

welded_beam_design_experiment = ComputerExperiment(
    _welded_beam_design_dict["input_dim"],
    _welded_beam_design_dict["input_box"],
    single_objective=_welded_beam_design_dict["single_objective"],
    single_constraint=_welded_beam_design_dict["single_constraint"],
    constraint_bounds=_welded_beam_design_dict["constraint_bounds"]
)


# ===== SRN =====
def _srn_objectives(x):
    x1, x2 = x.T
    f1 = 2 + (x1 - 2) ** 2 + (x2 - 2) ** 2
    f2 = 9 * x1 ** 2 - (x2 - 1) ** 2
    return np.column_stack((f1, f2))

def _srn_constraints(x):
    x1, x2 = x.T
    c1 = x1 ** 2 + x2 ** 2 - 225
    c2 = x1 - 3 * x2 + 10
    return np.column_stack((c1, c2))

_srn_dict = {
    "input_dim": 2,
    "input_box": [[-20, -20], [20, 20]],
    "single_objective": {'function': _srn_objectives, 'output_dim': 2},
    "single_constraint": {'function': _srn_constraints, 'output_dim': 2},
    "constraint_bounds": [[-np.inf, 0]] * 2
}

srn_experiment = ComputerExperiment(
    _srn_dict["input_dim"],
    _srn_dict["input_box"],
    single_objective=_srn_dict["single_objective"],
    single_constraint=_srn_dict["single_constraint"],
    constraint_bounds=_srn_dict["constraint_bounds"]
)

# ===== TNK Problem ======
def _tnk_objectives(x):
    x1, x2 = x.T
    f1 = x1
    f2 = x2
    return np.column_stack((f1, f2))

def _tnk_constraints(x):
    x1, x2 = x.T
    c1 = np.where((x1 == 0) & (x2 == 0), 1, 1 + 0.1 * np.cos(16 * np.arctan(x1/x2)) - x1 ** 2 - x2 ** 2)
    c2 = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 - 0.5
    return np.column_stack((c1, c2))

_tnk_dict = {
    "input_dim": 2,
    "input_box": [[0, 0], [np.pi, np.pi]],
    "single_objective": {'function': _tnk_objectives, 'output_dim': 2},
    "single_constraint": {'function': _tnk_constraints, 'output_dim': 2},
    "constraint_bounds": [[-np.inf, 0]] * 2
}

tnk_experiment = ComputerExperiment(
    _tnk_dict["input_dim"],
    _tnk_dict["input_box"],
    single_objective=_tnk_dict["single_objective"],
    single_constraint=_tnk_dict["single_constraint"],
    constraint_bounds=_tnk_dict["constraint_bounds"]
)

# TwoBarTruss Problem
def _two_bar_truss_objectives(x):
    x1, x2, x3 = x.T
    sig1 = 20 * np.sqrt(16 + x3 ** 2) / x1 / x3
    sig2 = 80 * np.sqrt(1 + x3 ** 2) / x2 / x3
    f1 = x1 * np.sqrt(16 + x3 ** 2) + x2 * np.sqrt(1 + x3 ** 2)
    f2 = np.maximum(sig1, sig2)
    return np.column_stack((f1, f2))

def _two_bar_truss_constraints(x):
    x1, x2, x3 = x.T
    sig1 = 20 * np.sqrt(16 + x3 ** 2) / x1 / x3
    sig2 = 80 * np.sqrt(1 + x3 ** 2) / x2 / x3
    c1 = np.maximum(sig1, sig2) - 10**5
    return c1.reshape(-1, 1)

_two_bar_truss_dict = {
    "input_dim": 3,
    "input_box": [[1e-12, 1e-12, 1], [0.01, 0.01, 3]],
    "single_objective": {'function': _two_bar_truss_objectives, 'output_dim': 2},
    "single_constraint": {'function': _two_bar_truss_constraints, 'output_dim': 1},
    "constraint_bounds": [[-np.inf, 0]]
}

two_bar_truss_experiment = ComputerExperiment(
    _two_bar_truss_dict["input_dim"],
    _two_bar_truss_dict["input_box"],
    single_objective=_two_bar_truss_dict["single_objective"],
    single_constraint=_two_bar_truss_dict["single_constraint"],
    constraint_bounds=_two_bar_truss_dict["constraint_bounds"]
)

# CONSTR Problem
def _constr_objectives(x):
    x1, x2 = x.T
    f1 = x1
    f2 = (1 + x2) / x1
    return np.column_stack((f1, f2))

def _constr_constraints(x):
    x1, x2 = x.T
    c1 = 6 - x2 - 9 * x1
    c2 = 1 + x2 - 9 * x1
    return np.column_stack((c1, c2))

_constr_dict = {
    "input_dim": 2,
    "input_box": [[0.1, 0], [1, 5]],
    "single_objective": {'function': _constr_objectives, 'output_dim': 2},
    "single_constraint": {'function': _constr_constraints, 'output_dim': 2},
    "constraint_bounds": [[-np.inf, 0]] * 2
}

constr_experiment = ComputerExperiment(
    _constr_dict["input_dim"],
    _constr_dict["input_box"],
    single_objective=_constr_dict["single_objective"],
    single_constraint=_constr_dict["single_constraint"],
    constraint_bounds=_constr_dict["constraint_bounds"]
)

# ===== WeldedBeamDesignSingleObj =====

def _welded_beam_objectives(x):
    h, l, t, b = x.T
    return 1.10471 * l * h**2 + 0.04811 * t * b * (14 + l)

def _welded_beam_constraints(x):
    h, l, t, b = x.T
    tau_max = 13600
    sigma_max = 30000
    pc_min = 6000
    delta_max = 0.25

    tau_prime = 6000 / (np.sqrt(2) * h * l)
    tau_second = (6000 * (14 + 0.5 * l) * np.sqrt(0.25 * (l**2 + (h + t)**2))) / (2 * np.sqrt(2) * h * l * ((l**2) / 12 + 0.25 * (h + t)**2))
    tau = np.sqrt(tau_prime**2 + tau_second**2 + (l * tau_prime * tau_second) / np.sqrt(0.25 * (l**2 + (h + t)**2)))
    
    sigma = 504000 / (b * t**2)

    pc = 64746.022 * (1 - 0.0282346 * t) * t * b**3

    c1 = -(tau_max - tau)
    c2 = -(sigma_max - sigma)
    c3 = -(b - h)
    c4 = (0.10471 * h**2 + 0.04811 * t * b * (14 + l) - 5) / 5
    c5 = (2.1952 / (b * t**3) - delta_max) / delta_max
    c6 = (pc_min - pc) / pc_min

    return np.column_stack((c1, c2, c3, c4, c5, c6))

_welded_beam_dict = {
    "input_dim": 4,
    "input_box": [[0.125, 0.1, 0.1, 0.1], [10, 10, 10, 10]],
    "single_objective": _welded_beam_objectives,
    "single_constraint": {'function': _welded_beam_constraints, 'output_dim': 6},
    "constraint_bounds": [[-np.inf, 0]] * 6
}

welded_beam_experiment = ComputerExperiment(
    _welded_beam_dict["input_dim"],
    _welded_beam_dict["input_box"],
    single_objective=_welded_beam_dict["single_objective"],
    single_constraint=_welded_beam_dict["single_constraint"],
    constraint_bounds=_welded_beam_dict["constraint_bounds"]
)

# ===== OSY =====

def _osy_objectives(x):
    x1, x2, x3, x4, x5, x6 = x.T
    f1 = -25*(x1-2)**2 - (x2-2)**2 - (x3-1)**2 - (x4-4)**2 - (x5-1)**2
    f2 = np.sum(x**2, axis = 1)
    return np.column_stack((f1, f2))

def _osy_constraints(x):
    x1, x2, x3, x4, x5, x6 = x.T
    c1 = x1 + x2 - 2
    c2 = 6 - x1 - x2
    c3 = 2 - x2 + x1
    c4 = 2 - x1 + 3*x2
    c5 = 4 - (x3-3)**2 - x4
    c6 = (x5-3)**2 + x6 - 4
    return np.column_stack((-c1, -c2, -c3, -c4, -c5, -c6))

_osy_dict = {
    "input_dim": 6,
    "input_box": [[0, 0, 1, 0, 1, 0], [10, 10, 5, 6, 5, 10]],
    "single_objective": _osy_objectives,
    "single_constraint": {'function': _osy_constraints, 'output_dim': 6},
    "constraint_bounds": [[-np.inf, 0]] * 6
}

osy_experiment = ComputerExperiment(
    _osy_dict["input_dim"],
    _osy_dict["input_box"],
    single_objective=_osy_dict["single_objective"],
    single_constraint=_osy_dict["single_constraint"],
    constraint_bounds=_osy_dict["constraint_bounds"]
)


# ===== SIN2 ======
def _sin2_objectives(x):
    x1, x2 = x.T
    return -(np.sin(13 * x1) * np.sin(27 * x1) + 1) * (np.sin(13 * x2) * np.sin(27 * x2) + 1) / 2

_sin2_dict = {
    "input_dim": 2,
    "input_box": [[0, 0], [1, 1]],
    "single_objective": _sin2_objectives,
}

sin2_experiment = ComputerExperiment(
    _sin2_dict["input_dim"],
    _sin2_dict["input_box"],
    single_objective=_sin2_dict["single_objective"],
)


# ====== ACKLEY ======
def create_ackley_problem(d):
    def _ackley_objectives(x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        return - a * np.exp(- b * np.sqrt((x**2).mean(1))) - np.exp(np.cos(c * x).mean(1)) + a + np.exp(1)

    _ackley_dict = {
        "input_dim": d,
        "input_box": [[-32.768]*d, [32.768]*d],
        "single_objective": _ackley_objectives,
    }

    ackley_experiment = ComputerExperiment(
        _ackley_dict["input_dim"],
        _ackley_dict["input_box"],
        single_objective=_ackley_dict["single_objective"],
    )

    return ackley_experiment

ackley4 = create_ackley_problem(4)
ackley6 = create_ackley_problem(6)
ackley10 = create_ackley_problem(10)


# ===== Rastrigin =====
def create_rastrigin_problem(d):
    def _rastrigin_objectives(x):
        return 10 * d + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum(1)

    _rastrigin_dict = {
        "input_dim": d,
        "input_box": [[-5.12]*d, [5.12]*d],
        "single_objective": _rastrigin_objectives,
    }

    rastrigin_experiment = ComputerExperiment(
        _rastrigin_dict["input_dim"],
        _rastrigin_dict["input_box"],
        single_objective=_rastrigin_dict["single_objective"],
    )
    return rastrigin_experiment

rastrigin10 = create_rastrigin_problem(10)


# ===== Rosenbrock =====

def create_rosenbrock_problem(d):
    def _rosenbrock_objectives(x):
        obj = 0
        for i in range(d - 1):
            obj += 100 * (x[:, i + 1] - x[:, i] ** 2) ** 2 + (x[:, i] - 1) ** 2
        return obj

    _rosenbrock_dict = {
        "input_dim": d,
        "input_box": [[-5]*d, [10]*d],
        "single_objective": _rosenbrock_objectives,
    }

    rosenbrock_experiment = ComputerExperiment(
        _rosenbrock_dict["input_dim"],
        _rosenbrock_dict["input_box"],
        single_objective=_rosenbrock_dict["single_objective"],
    )

    return rosenbrock_experiment

rosenbrock4 = create_rosenbrock_problem(4)
rosenbrock6 = create_rosenbrock_problem(6)
rosenbrock10 = create_rosenbrock_problem(10)


# ===== Schwefel =====
def create_schwefel_problem(d):
    def _schwefel_objectives(x):
        return 418.9829 * d - (x * np.sin(np.sqrt(np.abs(x)))).sum(1)

    _schwefel_dict = {
        "input_dim": d,
        "input_box": [[-500]*d, [500]*d],
        "single_objective": _schwefel_objectives,
    }

    schwefel = ComputerExperiment(
        _schwefel_dict["input_dim"],
        _schwefel_dict["input_box"],
        single_objective=_schwefel_dict["single_objective"],
    )

    return schwefel

schwefel10 = create_schwefel_problem(10)

# ===== ThreeHumpCamelBack ======

def _three_hump_camel_back_objectives(x):
    x1, x2 = x.T
    return 2 * x1 ** 2 - 1.05 * x1**4 + (x1**6)/6 + x1 * x2 + x2**2

_three_hump_camel_back_dict = {
    "input_dim": 2,
    "input_box": [[-5, -5], [5, 5]],
    "single_objective": _three_hump_camel_back_objectives,
}

threehumpcamelback = ComputerExperiment(
    _three_hump_camel_back_dict["input_dim"],
    _three_hump_camel_back_dict["input_box"],
    single_objective=_three_hump_camel_back_dict["single_objective"],
)


# ===== CamelBack =====
def _camel_back_objectives(x):
    x1, x2 = x.T
    y = - (4 - 2.1 * x1 ** 2 + (x1 ** 4)/3) * x1 ** 2 - x1 * x2 - (- 4 + 4 * x2 ** 2) * x2 ** 2
    return - y

_camel_back_dict = {
    "input_dim": 2,
    "input_box": [[-3, -2], [3, 2]],
    "single_objective": _camel_back_objectives,
}

camel_back = ComputerExperiment(
    _camel_back_dict["input_dim"],
    _camel_back_dict["input_box"],
    single_objective=_camel_back_dict["single_objective"],
)


# ===== Shubert =====
def _shubert_objectives(x):
    x1, x2 = x.T
    tmp1 = sum(i * np.cos((i + 1) * x1 + i) for i in range(1, 6))
    tmp2 = sum(i * np.cos((i + 1) * x2 + i) for i in range(1, 6))
    return tmp1 * tmp2

_shubert_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "single_objective": _shubert_objectives,
}

shubert = ComputerExperiment(
    _shubert_dict["input_dim"],
    _shubert_dict["input_box"],
    single_objective=_shubert_dict["single_objective"],
)

# ===== Dixon Price  ======
# See https://www.sfu.ca/~ssurjano/dixonpr.html

def create_dixon_price_problem(d):
    def _dixon_price_objectives(x):
        y = (x[:, 0] - 1) ** 2
        for i in range(1, d):
            y += (i + 1) * (2 * x[:, i]**2 - x[:, i - 1])**2
        return y

    _dixon_price_dict = {
        "input_dim": d,
        "input_box": [[-10]*d, [10]*d],
        "single_objective": _dixon_price_objectives,
    }

    dixon_price_experiment = ComputerExperiment(
        _dixon_price_dict["input_dim"],
        _dixon_price_dict["input_box"],
        single_objective=_dixon_price_dict["single_objective"],
    )

    return dixon_price_experiment

dixon_price4 = create_dixon_price_problem(4)
dixon_price6 = create_dixon_price_problem(6)
dixon_price10 = create_dixon_price_problem(10)

# =====
def create_trid_problem(d):
    def _trid_objective(x):
        y = ((x - 1) ** 2).sum(axis=1)

        for i in range(1, x.shape[1]):
            y = y - x[:, i] * x[:, i - 1]

        return y

    _trid_dict = {
        "input_dim": d,
        "input_box": [[-10**2]*d, [10**2]*d],
        "single_objective": _trid_objective
    }

    trid_experiment = ComputerExperiment(
        _trid_dict["input_dim"],
        _trid_dict["input_box"],
        single_objective=_trid_dict["single_objective"],
    )

    return trid_experiment

trid10 = create_trid_problem(10)

# ==== PERM

def create_perm_problem(d):
    def _perm_objective(x, beta=1):
        y = 0

        for i in range(x.shape[1]):
            tmp = 0
            for j in range(x.shape[1]):
                tmp = tmp + ((j + 1) + beta) * (x[:, j] ** (i + 1) - 1/(j + 1)**(i + 1))

            y = y + (tmp ** 2)

        return y

    _perm_dict = {
        "input_dim": d,
        "input_box": [[-d]*d, [d]*d],
        "single_objective": lambda x: _perm_objective(x, beta=1.0)
    }

    perm_experiment = ComputerExperiment(
        _perm_dict["input_dim"],
        _perm_dict["input_box"],
        single_objective=_perm_dict["single_objective"],
    )

    return perm_experiment

perm4 = create_perm_problem(4)
perm6 = create_perm_problem(6)
perm10 = create_perm_problem(10)

# ==== Michalewicz

def create_michalewicz_problem(d):
    
    def _michalewicz_objective(x, m=10):
        i_table = np.tile(np.arange(0, x.shape[1]) + 1, reps=[x.shape[0], 1])

        y = - np.sin(x) * np.sin(i_table * x ** 2 /np.pi) ** (2 * m)

        return y.sum(axis=1)

    _michalewicz_dict = {
        "input_dim": d,
        "input_box": [[0]*d, [np.pi]*d],
        "single_objective": lambda x: _michalewicz_objective(x, m=10)  # Use your desired m
    }

    michalewicz_experiment = ComputerExperiment(
        _michalewicz_dict["input_dim"],
        _michalewicz_dict["input_box"],
        single_objective=_michalewicz_dict["single_objective"],
    )

    return michalewicz_experiment

michalewicz4 = create_michalewicz_problem(4)
michalewicz6 = create_michalewicz_problem(6)
michalewicz10 = create_michalewicz_problem(10)

# ==== Zakharov

def create_zakharov_problem(d):

    def _zakharov_objective(x):
        i_table = np.tile(np.arange(0, d) + 1, reps=[x.shape[0], 1])
        sum_i = (0.5 * i_table * x).sum(1)
        return (x ** 2).sum(1) + sum_i**2 + sum_i**4

    _zakharov_dict = {
        "input_dim": d,
        "input_box": [[-5]*d, [10]*d],
        "single_objective": _zakharov_objective,
    }

    zakharov = ComputerExperiment(
        _zakharov_dict["input_dim"],
        _zakharov_dict["input_box"],
        single_objective=_zakharov_dict["single_objective"]
    )

    return zakharov

zakharov4 = create_zakharov_problem(4)
zakharov6 = create_zakharov_problem(6)
zakharov10 = create_zakharov_problem(10)

# ==== Easom
def _easom_objective(x):
    return - np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.exp(- (x[:, 0] - np.pi)**2 - (x[:, 1] - np.pi)**2)

_easom_dict = {
    "input_dim": 2,
    "input_box": [[-100, -100], [100, 100]],
    "single_objective": _easom_objective,
}

easom = ComputerExperiment(
    _easom_dict["input_dim"],
    _easom_dict["input_box"],
    single_objective=_easom_dict["single_objective"]
)

# ===== Matyas
def _matyas_objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2

_matyas_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "single_objective": _matyas_objective,
}

matyas = ComputerExperiment(
    _matyas_dict["input_dim"],
    _matyas_dict["input_box"],
    single_objective=_matyas_dict["single_objective"]
)


# ===== BOOTH

def _booth_objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2

_booth_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "single_objective": _booth_objective,
}

booth = ComputerExperiment(
    _booth_dict["input_dim"],
    _booth_dict["input_box"],
    single_objective=_booth_dict["single_objective"]
)


# ===== CrossInTray

def _crossintray_objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return - 10**(-4) * (
        np.abs(
            np.sin(x1) * np.sin(x2) * np.exp(np.abs(100 - np.sqrt(x1 ** 2 + x2 ** 2)/np.pi))
        ) + 1
    ) ** (0.1)

_crossintray_dict = {
    "input_dim": 2,
    "input_box": [[-10, -10], [10, 10]],
    "single_objective": _crossintray_objective,
}

crossintray = ComputerExperiment(
    _crossintray_dict["input_dim"],
    _crossintray_dict["input_box"],
    single_objective=_crossintray_dict["single_objective"]
)

# ====== CrossInTrayZoom

_crossintrayzoom_dict = {
    "input_dim": 2,
    "input_box": [[-2, -2], [2, 2]],
    "single_objective": _crossintray_objective,
}

crossintrayzoom = ComputerExperiment(
    _crossintrayzoom_dict["input_dim"],
    _crossintrayzoom_dict["input_box"],
    single_objective=_crossintrayzoom_dict["single_objective"]
)

# ===== Beale

def _beale_objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2

_beale_dict = {
    "input_dim": 2,
    "input_box": [[-4.5, -4.5], [4.5, 4.5]],
    "single_objective": _beale_objective,
}

beale = ComputerExperiment(
    _beale_dict["input_dim"],
    _beale_dict["input_box"],
    single_objective=_beale_dict["single_objective"]
)


# ===== Branin
def _branin_objective(x):
    a = 1
    b = 5.1 / (4 * (np.pi) ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2 + s * (1 - t) * np.cos(x[:, 0]) + s

_branin_dict = {
    "input_dim": 2,
    "input_box": [[-5, 0], [10, 15]],
    "single_objective": _branin_objective,
}

branin = ComputerExperiment(
    _branin_dict["input_dim"],
    _branin_dict["input_box"],
    single_objective=_branin_dict["single_objective"]
)

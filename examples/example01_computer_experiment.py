"""Example Script for ComputerExperiment Class

This script demonstrates the usage of the ComputerExperiment class
from the computerexperiment module.  The ComputerExperiment class is
designed for specifying and evaluating functions in a computer
experiment problem. Additionally, functions can be labeled as
objectives or constraints.

The script provides two examples:
1. Using separate objective and constraint functions.
2. Using a combined function that handles both objectives and
   constraints.

Each  example demonstrates  how to  initialize the  ComputerExperiment
object, how to evaluate functions, and how to handle constraints.

Author: Emmanuel Vazquez <emmanuel.vazquez@centralesupelec.fr>
Copyright (c) 2022-2023, CentraleSupelec
License: GPLv3 (see LICENSE)

"""

import numpy as np
import gpmp as gp
import gpmpcontrib as gpc


# Define an objective function
def _pb_objective(x):
    """Objective function: calculates a specific mathematical operation."""
    return (x[:, 0] - 10) ** 3 + (x[:, 1] - 20) ** 3


# Define two constraint functions
def _pb_constraints(x):
    """Constraint functions: calculates constraints based on input conditions."""
    c1 = -((x[:, 0] - 5) ** 2) - (x[:, 1] - 5) ** 2 + 100
    c2 = (x[:, 0] - 6) ** 2 + (x[:, 1] - 5) ** 2 - 82.81
    return np.column_stack((c1, c2))


# Set up parameters for the ComputerExperiment class
_pb_dict = {
    "input_dim": 2,
    "input_box": [[13, 0], [100, 100]],
    "single_objective": _pb_objective,
    "single_constraint": {
        "function": _pb_constraints,
        "output_dim": 2,
        "bounds": [[100.0, np.inf], [-np.inf, 82.81]],
    },
}

# ------------------------------------------------------------
# Example 1: Using separate objective and constraint functions

print("  *** Example 1")
print("      =========")
# Initialize the ComputerExperiment object with separate objective and constraint
pb = gpc.ComputerExperiment(
    _pb_dict["input_dim"],
    _pb_dict["input_box"],
    single_objective=_pb_dict["single_objective"],
    single_constraint=_pb_dict["single_constraint"],
)

# Display the initialized ComputerExperiment object.
# This print statement shows the configuration of the experiment, including input
# dimensions, objectives, and constraints.
print(pb)

# Test the evaluation of the objective and constraint functions
x = np.array([[50.0, 50.0], [80.0, 80.0]])
print("\n * Input X:\n", x)

# Evaluate the function using the __call__ method and print the results
results = pb(x)
print("\n * Evaluated Results (one objective, two constraints):\n", results)

# Evaluate the objectives and print the results.
# Note: This uses the previous computation since x is unchanged
results_objective = pb.eval_objectives(x)
print("\n * Objectives only:\n", results_objective)

# Evaluate the constraints and print the results.
# Note: This uses the previous computation since x is unchanged
results_constraint = pb.eval_constraints(x)
print("\n * Constraints only:\n", results_constraint)

# ------------------------------------------------------------------------
# Example 2: Using a combined function for both objectives and constraints
print("\n")
print("  *** Example 2")
print("      =========")


# Define a combined evaluation function
def _pb_evaluation(x):
    """Combined function for evaluating both objectives and constraints."""
    return np.column_stack((_pb_objective(x), _pb_constraints(x)))


# Initialize the ComputerExperiment object with the combined function
pb = gpc.ComputerExperiment(
    _pb_dict["input_dim"],
    _pb_dict["input_box"],
    single_function={
        "function": _pb_evaluation,
        "output_dim": 1 + 2,
        "type": ["objective"] + ["constraint"] * 2,
        "bounds": [None] + [[100.0, np.inf], [-np.inf, 82.81]],
    },
)

# Display the initialized ComputerExperiment object with the combined function.
# This configuration is an alternative approach where a single
# function handles both objectives and constraints.
print(" " * 2 + str(pb).replace("\n", "\n  "))

# Test the evaluation of the combined function
results = pb(x)
print("\n  * Evaluated Results (one objective, two constraints):\n", results)
results_objective = pb.eval_objectives(x)
print("\n  * Objectives only:\n", results_objective)

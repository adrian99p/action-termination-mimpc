"""Action-based termination of branch-and-bound for mixed-integer MPC.

    problem.py          MIQPProblem, the MIQP of one sampling instant
    action_encoding.py  ActionEncoding, binary patterns to first inputs
    status.py           SolveStatus, shared by all solvers
    bnb/                branch-and-bound with action termination and action-first branching
    oracle/             full and action-fixed Gurobi MIQP solves, used as references
    gurobi/             Gurobi environment and C API access for the node QPs
    benchmarks/         the four benchmark problems
    mpc/                closed-loop simulation
    experiment/         experiment runner, result files, statistics, and output helpers
"""

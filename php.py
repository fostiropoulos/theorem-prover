"""
Evaluate the performance of Z3 on the Pigeonhole Principle (PHP).
PHP is a known unsatisfiable problem where n pigeons have to be fit
to n-1 holes. Resolution is usually slow for small n. In this module
we obtain a speed improvement in resolution by encoding each pigeon as
a boolean variable and summing them up.
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from z3 import Bool, If, Solver, Sum, sat


def php_solver(n: int, is_sat=False) -> float:
    """php_solver evaluates the time performance of z3 on the
    Pigeonhole (PHP) principle. The encoding is meant to be efficient, for the solver.

    Parameters
    ----------
    n : int
        the number of pigeons, must be positive
    is_sat : bool
        whether the PHP should be sat (i.e. holes (m) == n)
        or unsat i.e. m=n-1
    Returns
    -------
    float
        duration in seconds for unsat
    """
    sol = Solver()
    m = n
    if not is_sat:
        m -= 1

    # variables
    p = {}
    for i in range(n):
        for j in range(m):
            # Represent each pigeon-hole as a boolean i.e. 0-1 integer
            p[(i, j)] = Bool(f"({i:d},{j:d})")

    # One pigeon per hole
    for j in range(m):
        sol.add(Sum([If(p[(i, j)], 1, 0) for i in range(n)]) <= 1)

    # Every pigeon in one hole
    for i in range(n):
        sol.add(Sum([If(p[(i, j)], 1, 0) for j in range(m)]) == 1)

    start = time.time()
    res = sol.check()
    end = time.time()
    # sanity checks
    if not is_sat:
        assert res != sat
    else:
        assert res == sat
    duration = end - start
    return duration


def make_fig(save_dir: Path, durs: np.ndarray):
    """make_fig creates report figures

    Parameters
    ----------
    save_dir : Path
        the directory to save the evaluation figure
    durs : np.ndarray
        an array of the duration results
    """
    fig, axs = plt.subplots(1, 1)
    x = np.arange(2, durs.shape[0] + 2)
    y = durs.mean(-1)
    axs.plot(x, y,label="Int. Encoding PHP")
    error = durs.std(-1)
    axs.fill_between(x, y - error, y + error, alpha=0.5)
    axs.set_ylabel("Solving Duration (s)")
    axs.set_xlabel("N-Pigeons N-1 Holes")
    axs.set_title("Solving time of Pigeonhole Principle using Z3")
    axs.legend()
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir.joinpath("evaluation.png"))


def evaluate_performance(save_dir: Path = Path("assets"), n_range=100, repetitions=10):
    """evaluate_performance evaluates the performance of z3 on the PHP principle

    Parameters
    ----------
    save_dir : Path, optional
        the directory to save the evaluation figure, by default Path("assets")
    n_range : int, optional
        the maximum n to test, by default 100
    repetitions : int, optional
        the number of repetitions for each n, by default 10
    """
    durations = []
    for i in range(2, n_range):
        results_i = []
        for j in range(repetitions):
            results_i.append(php_solver(i))
        durations.append(results_i)
    durs = np.array(durations)
    make_fig(save_dir, durs)
    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser("Pigeonhole Principle Z3 performance evaluation")
    args.add_argument("--save-dir", type=Path, default=Path("assets"))
    args.add_argument(
        "--n-range", type=int, help="the maximum n to test", default=100
    )
    args.add_argument(
        "--repetitions",
        type=int,
        help="the number of repetitions for each n",
        default=10,
    )
    evaluate_performance(**vars(args.parse_args()))

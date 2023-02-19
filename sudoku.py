"""
Module for solving Sudoku 9x9 puzzles using Z3.
"""
import fileinput
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from z3 import Distinct, Int, Solver, sat


def sudoku_solver(sudoku: List[str]) -> str:
    """sudoku_solver solve the given sudoku using Z3

    Parameters
    ----------
    sudoku : List[str]
        List of 9 strings of length 9

    Returns
    -------
    str
        the solution string formatted with new line characters
        at every 3 characters.
    """
    first_row = sudoku[0]
    # Only works on 9x9 (can be easily extended to general cases)
    assert (
        (np.array([len(r) for r in sudoku]) == len(first_row)).all()
        and len(first_row) == 9
        and len(sudoku) == 9
    )
    x = np.arange(len(sudoku))
    y = np.arange(len(first_row))
    s = Solver()
    board: Dict[Tuple[int, int], Int] = {}
    # board definition
    for i in x:
        for j in y:
            board[(i, j)] = Int(f"({i:d},{j:d})")
            s.add(board[(i, j)] <= 9, board[(i, j)] >= 1)

    for i in x:
        # distinct row
        s.add(Distinct([board[(i, j)] for j in y]))
        # distinct col
        s.add(Distinct([board[(j, i)] for j in y]))

    # distinct grid
    for i in range(3):
        for j in range(3):
            s.add(
                Distinct(
                    [board[(m + i * 3, n + j * 3)] for m in range(3) for n in range(3)]
                )
            )
    # now we put the assumptions of the given puzzle into the solver:
    for i in x:
        for j in y:
            if sudoku[i][j] != " " and sudoku[i][j] != ".":
                s.add(board[(i, j)] == int(sudoku[i][j]))

    assert s.check() == sat, "Unsat"

    model = s.model()
    solution = "\n".join(
        ["".join([model.evaluate(board[(i, j)]).as_string() for j in y]) for i in x]
    )
    return solution


def std_input_to_sudoku() -> List[str]:
    """std_input_to_sudoku read a sudoku puzzle from std-in

    Returns
    -------
    List[str]
        the processed sudoku puzzle
    """
    sudoku = []
    for line in fileinput.input():
        line = line.rstrip("\n")
        assert len(line) == 9, "Invalid length input. Can only solve 9x9."
        sudoku.append(line)
        if len(sudoku)==9:
            break
    return sudoku


def main():
    """main solve a sudoku from stdin. Can use '.' or ' ' for empty cells.
    i.e.
...2857..
.193.....
.8...1.6.
.45.6....
.27...14.
....5.68.
.3.9...5.
.....347.
..1528...
    """
    puzzle = std_input_to_sudoku()
    s_solved = sudoku_solver(puzzle)
    print(s_solved)


def test_validity():
    """test_validity test validity of the algorithm on a sudoku dataset.
        100 sudoku sampled from:
        https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings
    """
    df = pd.read_csv("assets/sudoku.csv")
    sudokus = df["puzzle"].apply(lambda x: np.array(list(x)).reshape(9, 9))
    solutions = sudokus.apply(lambda x: sudoku_solver(x).replace("\n", ""))
    assert (df["solution"] == solutions).all()


if __name__ == "__main__":
    main()

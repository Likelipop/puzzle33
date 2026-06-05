"""
solver.py — Backtracking solver for a 3×3 image patch puzzle.

The puzzle state is a list of 9 integers representing which original patch
occupies each of the 9 grid positions (row-major order):

    Position layout:
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8

    Goal state: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    (every patch is in its correct position)

Allowed moves: swap any two *adjacent* patches (horizontal or vertical neighbour).
The solver uses iterative-deepening DFS (= backtracking with a depth limit)
plus a simple Manhattan-distance heuristic to order candidate moves.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Adjacency map — which positions can be swapped?
# ---------------------------------------------------------------------------

ADJACENCY: dict[int, list[int]] = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7],
}

GOAL: Tuple[int, ...] = tuple(range(9))
GRID_SIZE = 3


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan_distance(state: Tuple[int, ...]) -> int:
    """Sum of Manhattan distances of each tile from its goal position."""
    total = 0
    for pos, tile in enumerate(state):
        goal_row, goal_col = divmod(tile, GRID_SIZE)
        cur_row,  cur_col  = divmod(pos,  GRID_SIZE)
        total += abs(goal_row - cur_row) + abs(goal_col - cur_col)
    return total


def _swap(state: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    lst = list(state)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def _sorted_neighbors(state: Tuple[int, ...], pos: int) -> List[int]:
    """Return neighbours of *pos* sorted so the swap reducing h(state) comes first."""
    return sorted(
        ADJACENCY[pos],
        key=lambda nb: manhattan_distance(_swap(state, pos, nb)),
    )


# ---------------------------------------------------------------------------
# Step dataclass
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One move in the solution sequence."""
    state:       Tuple[int, ...]
    move_from:   int
    move_to:     int
    step_number: int
    description: str = field(init=False)

    def __post_init__(self):
        fr, fc = divmod(self.move_from, GRID_SIZE)
        tr, tc = divmod(self.move_to,   GRID_SIZE)
        self.description = (
            f"Step {self.step_number}: swap patch at "
            f"row {fr+1}, col {fc+1}  ↔  row {tr+1}, col {tc+1}"
        )


# ---------------------------------------------------------------------------
# Iterative-deepening DFS (backtracking)
# ---------------------------------------------------------------------------

def _dfs(
    state:     Tuple[int, ...],
    path:      List[Tuple[int, int]],
    visited:   set,
    depth:     int,
    max_depth: int,
) -> Optional[List[Tuple[int, int]]]:
    if state == GOAL:
        return path
    if depth == max_depth:
        return None

    for pos in range(9):
        for nb in _sorted_neighbors(state, pos):
            next_state = _swap(state, pos, nb)
            if next_state in visited:
                continue
            visited.add(next_state)
            result = _dfs(next_state, path + [(pos, nb)], visited, depth + 1, max_depth)
            if result is not None:
                return result
            visited.discard(next_state)

    return None


def solve(
    initial_state: List[int] | Tuple[int, ...],
    max_depth: int = 25,
) -> List[Step]:
    """
    Args:
        initial_state: length-9 sequence where initial_state[pos] = tile index.
        max_depth:     Maximum swaps before giving up.
    Returns:
        Ordered list of Step objects. Empty list if already solved.
    Raises:
        NoSolutionError if no solution found within max_depth.
    """
    state = tuple(initial_state)
    if state == GOAL:
        return []

    visited = {state}
    for limit in range(1, max_depth + 1):
        swaps = _dfs(state, [], visited.copy(), 0, limit)
        if swaps is not None:
            return _build_steps(state, swaps)

    raise NoSolutionError(
        f"Could not solve the puzzle within {max_depth} moves. "
        "Try increasing max_depth or check the puzzle state."
    )


def _build_steps(initial: Tuple[int, ...], swaps: List[Tuple[int, int]]) -> List[Step]:
    steps: List[Step] = []
    state = initial
    for n, (i, j) in enumerate(swaps, start=1):
        state = _swap(state, i, j)
        steps.append(Step(state=state, move_from=i, move_to=j, step_number=n))
    return steps


def state_from_permutation(permutation: List[int]) -> Tuple[int, ...]:
    """
    Convert model output permutation to solver state.
    permutation[i] = j means shuffled patch i belongs at position j.
    The solver needs state[pos] = tile, which is the inverse permutation.
    """
    state = [0] * 9
    for shuffled_pos, target_pos in enumerate(permutation):
        state[target_pos] = shuffled_pos
    return tuple(state)


class NoSolutionError(Exception):
    pass
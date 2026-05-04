"""Tests for ``permute_people`` in :mod:`main`.

The key invariant being exercised here is that ``permute_people`` returns a
*permutation* of the input indices: every original index appears exactly once
in the output, with no duplicates and no missing values. The remaining tests
cover length, mutation behaviour, and that the input ``Table`` objects are not
modified in place.

Run with::

    pytest test_permute.py -v
"""

import numpy as np
import pytest

from main import Relationship, Table, flatten, permute_people


def make_tables(n_tables: int, table_size: int) -> list[Table]:
    """Create simple ``Table`` instances with empty relationships and sequential indices.

    ``Table`` is a frozen dataclass, so we construct fresh instances rather than
    mutating existing ones. Indices are assigned sequentially across tables so
    the full set of indices is ``range(n_tables * table_size)``.
    """
    tables: list[Table] = []
    for i in range(n_tables):
        start = i * table_size
        indices = list(range(start, start + table_size))
        people = [f"P{j}" for j in indices]
        tables.append(
            Table(relationships=[], people=people, indices=indices)
        )
    return tables


def original_indices(tables: list[Table]) -> np.ndarray:
    return np.array(flatten([t.indices for t in tables]))


def test_returns_valid_permutation():
    """Output indices are a permutation of the input indices."""
    tables = make_tables(n_tables=5, table_size=8)
    original = original_indices(tables)

    result = permute_people(tables, num_swaps=10)

    assert sorted(result.tolist()) == sorted(original.tolist()), (
        "permute_people must return a permutation of the original indices "
        "(no duplicates, no missing values)"
    )


def test_correct_length():
    """Output length equals the total number of people across all tables."""
    n_tables, table_size = 4, 6
    tables = make_tables(n_tables=n_tables, table_size=table_size)

    result = permute_people(tables, num_swaps=3)

    assert len(result) == n_tables * table_size


def test_swaps_actually_change_positions():
    """At least one position differs after non-zero swaps."""
    tables = make_tables(n_tables=5, table_size=8)
    original = original_indices(tables)

    result = permute_people(tables, num_swaps=5)

    assert not np.array_equal(result, original), (
        "permute_people with num_swaps>0 should change at least one position"
    )


def test_single_swap_changes_exactly_two_positions():
    """A single swap changes exactly 2 positions."""
    tables = make_tables(n_tables=5, table_size=8)
    original = original_indices(tables)

    result = permute_people(tables, num_swaps=1)

    diff_count = int(np.sum(result != original))
    assert diff_count == 2, (
        f"Expected exactly 2 differing positions for num_swaps=1, got {diff_count}"
    )


def test_n_swaps_change_at_most_2n_positions():
    """N swaps change at most 2N positions."""
    n_swaps = 7
    tables = make_tables(n_tables=5, table_size=8)
    original = original_indices(tables)

    result = permute_people(tables, num_swaps=n_swaps)

    diff_count = int(np.sum(result != original))
    assert diff_count <= 2 * n_swaps, (
        f"Expected at most {2 * n_swaps} differing positions for "
        f"num_swaps={n_swaps}, got {diff_count}"
    )


def test_input_not_mutated():
    """The original ``Table`` objects' indices are unchanged after permute_people."""
    tables = make_tables(n_tables=5, table_size=8)
    indices_before = [list(t.indices) for t in tables]

    permute_people(tables, num_swaps=10)

    indices_after = [list(t.indices) for t in tables]
    assert indices_before == indices_after, (
        "permute_people must not mutate the input Table objects' indices"
    )


def test_stress_random_iterations():
    """Run many random iterations to catch intermittent issues."""
    rng = np.random.default_rng(seed=12345)

    for iteration in range(200):
        n_tables = int(rng.integers(2, 8))
        table_size = int(rng.integers(2, 10))
        total = n_tables * table_size
        max_swaps = total // 2
        num_swaps = int(rng.integers(1, max_swaps + 1))

        tables = make_tables(n_tables=n_tables, table_size=table_size)
        original = original_indices(tables)
        indices_before = [list(t.indices) for t in tables]

        result = permute_people(tables, num_swaps=num_swaps)

        ctx = (
            f"iteration={iteration} n_tables={n_tables} "
            f"table_size={table_size} num_swaps={num_swaps}"
        )

        assert sorted(result.tolist()) == sorted(original.tolist()), (
            f"Result is not a permutation of the original indices ({ctx})"
        )
        assert len(result) == total, f"Wrong output length ({ctx})"

        diff_count = int(np.sum(result != original))
        assert diff_count <= 2 * num_swaps, (
            f"Expected at most {2 * num_swaps} differing positions, "
            f"got {diff_count} ({ctx})"
        )

        indices_after = [list(t.indices) for t in tables]
        assert indices_before == indices_after, (
            f"Input Table indices were mutated ({ctx})"
        )


def test_relationship_and_flatten_imports_are_usable():
    """Sanity check that the imported names are the expected types."""
    rel = Relationship(pair=("A", "B"), pair_is=(0, 1), score=1.0)
    assert rel.score == 1.0
    assert flatten([[1, 2], [3]]) == [1, 2, 3]

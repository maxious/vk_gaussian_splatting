"""Numba-optimized Union-Find (Disjoint Set) data structure.

Based on the fufpy library pattern (https://github.com/LuisScoccola/fufpy).
Uses path halving and union by rank for near-constant time operations.

Install dependencies:
    pip install numba numpy
"""

from __future__ import annotations

import logging

import numpy as np
from numba import jit

# Suppress verbose Numba debug logging
logging.getLogger("numba").setLevel(logging.WARNING)


@jit(nopython=True)
def union_find_init(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize Union-Find data structures.

    Args:
        n: Number of elements (0 to n-1)

    Returns:
        Tuple of (parent, rank) arrays
    """
    parent = np.arange(n, dtype=np.int_)
    rank = np.zeros(n, dtype=np.int_)
    return parent, rank


@jit(nopython=True)
def union_find_find(parent: np.ndarray, x: int) -> int:
    """Find the representative (root) of the set containing x.

    Uses path halving for amortized near-constant time.

    Args:
        parent: Parent array from union_find_init
        x: Element to find

    Returns:
        Representative element of the set containing x
    """
    while x != parent[x]:
        # Path halving: point to grandparent
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@jit(nopython=True)
def union_find_union(parent: np.ndarray, rank: np.ndarray, x: int, y: int) -> bool:
    """Merge the sets containing x and y.

    Uses union by rank for balanced trees.

    Args:
        parent: Parent array from union_find_init
        rank: Rank array from union_find_init
        x: First element
        y: Second element

    Returns:
        True if sets were merged, False if already in same set
    """
    xr = union_find_find(parent, x)
    yr = union_find_find(parent, y)

    if xr == yr:
        return False

    # Union by rank
    if rank[xr] > rank[yr]:
        parent[yr] = xr
    elif rank[xr] < rank[yr]:
        parent[xr] = yr
    else:
        parent[yr] = xr
        rank[xr] += 1

    return True


@jit(nopython=True)
def union_find_connected(parent: np.ndarray, x: int, y: int) -> bool:
    """Check if x and y are in the same set.

    Args:
        parent: Parent array from union_find_init
        x: First element
        y: Second element

    Returns:
        True if x and y are in the same set
    """
    return union_find_find(parent, x) == union_find_find(parent, y)


@jit(nopython=True)
def union_find_roots(parent: np.ndarray) -> np.ndarray:
    """Find all roots (one find per element).

    Args:
        parent: Parent array from union_find_init

    Returns:
        Array where roots[i] is the representative of element i
    """
    n = len(parent)
    roots = np.empty(n, dtype=np.int_)
    for i in range(n):
        roots[i] = union_find_find(parent, i)
    return roots


@jit(nopython=True)
def union_find_components(parent: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign sequential component IDs to all elements.

    Args:
        parent: Parent array from union_find_init

    Returns:
        Tuple of (component_ids, n_components) where:
        - component_ids[i] is the component ID for element i
        - n_components is the total number of unique components
    """
    n = len(parent)

    # Find all roots
    roots = np.empty(n, dtype=np.int_)
    for i in range(n):
        roots[i] = union_find_find(parent, i)

    # Assign sequential IDs
    component_ids = np.empty(n, dtype=np.int_)
    root_to_id = np.full(n, -1, dtype=np.int_)
    next_id = 0

    for i in range(n):
        root = roots[i]
        if root_to_id[root] == -1:
            root_to_id[root] = next_id
            next_id += 1
        component_ids[i] = root_to_id[root]

    return component_ids, next_id


class UnionFind:
    """Object-oriented wrapper for Union-Find operations.

    Example:
        >>> uf = UnionFind(10)
        >>> uf.union(0, 1)
        True
        >>> uf.union(1, 2)
        True
        >>> uf.connected(0, 2)
        True
        >>> uf.connected(0, 3)
        False
    """

    def __init__(self, n: int):
        """Create a Union-Find structure for n elements.

        Args:
            n: Number of elements (0 to n-1)
        """
        if n <= 0:
            raise ValueError("n must be positive")
        self._n = n
        self._parent, self._rank = union_find_init(n)

    def find(self, x: int) -> int:
        """Find the representative of the set containing x."""
        if x < 0 or x >= self._n:
            raise IndexError(f"Element {x} out of range [0, {self._n})")
        return union_find_find(self._parent, x)

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing x and y. Returns True if merged."""
        if x < 0 or x >= self._n:
            raise IndexError(f"Element {x} out of range [0, {self._n})")
        if y < 0 or y >= self._n:
            raise IndexError(f"Element {y} out of range [0, {self._n})")
        return union_find_union(self._parent, self._rank, x, y)

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        if x < 0 or x >= self._n:
            raise IndexError(f"Element {x} out of range [0, {self._n})")
        if y < 0 or y >= self._n:
            raise IndexError(f"Element {y} out of range [0, {self._n})")
        return union_find_connected(self._parent, x, y)

    def components(self) -> tuple[np.ndarray, int]:
        """Return (component_ids, n_components)."""
        return union_find_components(self._parent)

    @property
    def n_elements(self) -> int:
        """Number of elements in the structure."""
        return self._n


if __name__ == "__main__":
    # Quick demo
    import time

    print("Union-Find Demo")
    print("-" * 40)

    # Small example
    uf = UnionFind(10)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    print(f"connected(0, 2): {uf.connected(0, 2)}")  # True
    print(f"connected(0, 3): {uf.connected(0, 3)}")  # False

    ids, n = uf.components()
    print(f"Components: {n}")
    print(f"Component IDs: {ids}")

    # Performance test
    print("\nPerformance Test")
    print("-" * 40)

    n = 1_000_000
    print(f"Creating Union-Find with {n:,} elements...")

    # Warmup JIT
    parent, rank = union_find_init(100)
    union_find_union(parent, rank, 0, 1)
    union_find_find(parent, 0)

    start = time.perf_counter()
    parent, rank = union_find_init(n)
    init_time = time.perf_counter() - start
    print(f"  Init: {init_time * 1000:.2f}ms")

    # Union every pair (0,1), (2,3), ...
    start = time.perf_counter()
    for i in range(0, n - 1, 2):
        union_find_union(parent, rank, i, i + 1)
    union_time = time.perf_counter() - start
    print(f"  {n // 2:,} unions: {union_time * 1000:.2f}ms")

    # Find all
    start = time.perf_counter()
    roots = union_find_roots(parent)
    find_time = time.perf_counter() - start
    print(f"  {n:,} finds: {find_time * 1000:.2f}ms")

    # Components
    start = time.perf_counter()
    ids, n_comp = union_find_components(parent)
    comp_time = time.perf_counter() - start
    print(f"  Components ({n_comp:,}): {comp_time * 1000:.2f}ms")

"""Unit tests for Union-Find data structure."""

import numpy as np
import pytest

from offline.union_find import (
    UnionFind,
    union_find_components,
    union_find_connected,
    union_find_find,
    union_find_init,
    union_find_roots,
    union_find_union,
)


class TestUnionFindFunctions:
    """Test the low-level Numba-optimized functions."""

    def test_init(self):
        """Test initialization creates correct structure."""
        parent, rank = union_find_init(10)
        assert len(parent) == 10
        assert len(rank) == 10
        # Each element is its own parent initially
        assert np.array_equal(parent, np.arange(10))
        # All ranks start at 0
        assert np.all(rank == 0)

    def test_find_self(self):
        """Test find on unmerged elements returns self."""
        parent, rank = union_find_init(5)
        for i in range(5):
            assert union_find_find(parent, i) == i

    def test_union_basic(self):
        """Test basic union operation."""
        parent, rank = union_find_init(5)
        # Union 0 and 1
        result = union_find_union(parent, rank, 0, 1)
        assert result is True  # Merged
        # Now they should have the same root
        assert union_find_find(parent, 0) == union_find_find(parent, 1)

    def test_union_same_set(self):
        """Test union of elements already in same set returns False."""
        parent, rank = union_find_init(5)
        union_find_union(parent, rank, 0, 1)
        # Try to union again
        result = union_find_union(parent, rank, 0, 1)
        assert result is False  # Already in same set

    def test_union_transitivity(self):
        """Test that union is transitive."""
        parent, rank = union_find_init(5)
        union_find_union(parent, rank, 0, 1)
        union_find_union(parent, rank, 1, 2)
        # 0 and 2 should now be connected through 1
        assert union_find_connected(parent, 0, 2)

    def test_connected_basic(self):
        """Test connected check."""
        parent, rank = union_find_init(5)
        assert not union_find_connected(parent, 0, 1)
        union_find_union(parent, rank, 0, 1)
        assert union_find_connected(parent, 0, 1)
        assert not union_find_connected(parent, 0, 2)

    def test_roots(self):
        """Test getting all roots."""
        parent, rank = union_find_init(6)
        union_find_union(parent, rank, 0, 1)
        union_find_union(parent, rank, 2, 3)
        union_find_union(parent, rank, 3, 4)

        roots = union_find_roots(parent)
        # 0,1 should have same root
        assert roots[0] == roots[1]
        # 2,3,4 should have same root
        assert roots[2] == roots[3] == roots[4]
        # 5 is its own root
        assert roots[5] == 5
        # Groups should be different
        assert roots[0] != roots[2]
        assert roots[0] != roots[5]

    def test_components(self):
        """Test component assignment."""
        parent, rank = union_find_init(6)
        union_find_union(parent, rank, 0, 1)
        union_find_union(parent, rank, 2, 3)
        union_find_union(parent, rank, 3, 4)

        ids, n_components = union_find_components(parent)
        assert n_components == 3  # {0,1}, {2,3,4}, {5}
        # Same component elements have same ID
        assert ids[0] == ids[1]
        assert ids[2] == ids[3] == ids[4]
        # Different components have different IDs
        assert ids[0] != ids[2]
        assert ids[0] != ids[5]
        assert ids[2] != ids[5]


class TestUnionFindClass:
    """Test the object-oriented wrapper."""

    def test_init(self):
        """Test class initialization."""
        uf = UnionFind(10)
        assert uf.n_elements == 10

    def test_init_invalid(self):
        """Test that invalid sizes raise errors."""
        with pytest.raises(ValueError):
            UnionFind(0)
        with pytest.raises(ValueError):
            UnionFind(-1)

    def test_find(self):
        """Test find method."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_find_out_of_range(self):
        """Test find with out of range index."""
        uf = UnionFind(5)
        with pytest.raises(IndexError):
            uf.find(5)
        with pytest.raises(IndexError):
            uf.find(-1)

    def test_union(self):
        """Test union method."""
        uf = UnionFind(5)
        assert uf.union(0, 1) is True
        assert uf.union(0, 1) is False  # Already merged

    def test_connected(self):
        """Test connected method."""
        uf = UnionFind(5)
        assert not uf.connected(0, 1)
        uf.union(0, 1)
        assert uf.connected(0, 1)

    def test_components(self):
        """Test components method."""
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(3, 4)

        ids, n = uf.components()
        assert n == 3


class TestUnionFindPerformance:
    """Performance-focused tests."""

    def test_large_union_find(self):
        """Test with large number of elements."""
        n = 100_000
        parent, rank = union_find_init(n)

        # Union pairs
        for i in range(0, n - 1, 2):
            union_find_union(parent, rank, i, i + 1)

        # Verify
        ids, n_components = union_find_components(parent)
        assert n_components == n // 2

    def test_chain_unions(self):
        """Test chain of unions (worst case for naive implementation)."""
        n = 10_000
        parent, rank = union_find_init(n)

        # Chain: 0-1-2-3-...-(n-1)
        for i in range(n - 1):
            union_find_union(parent, rank, i, i + 1)

        # All should be in same component
        ids, n_components = union_find_components(parent)
        assert n_components == 1
        assert len(set(ids)) == 1

    def test_path_compression(self):
        """Verify path compression is working."""
        parent, rank = union_find_init(100)

        # Create deep chain
        for i in range(99):
            union_find_union(parent, rank, i, i + 1)

        # After find operations, paths should be compressed
        _ = union_find_find(parent, 0)
        _ = union_find_find(parent, 50)
        _ = union_find_find(parent, 99)

        # Root should be reachable in few hops now
        # (path halving doesn't fully flatten, but reduces depth)
        root = union_find_find(parent, 0)
        hops = 0
        x = 0
        while parent[x] != x:
            x = parent[x]
            hops += 1
        # Should be significantly less than 99 due to compression
        assert hops < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

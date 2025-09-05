from __future__ import annotations

import numpy as np
import pytest

from constelx.physics.pbfm import conflict_free_update


def test_conflict_free_update_non_conflicting() -> None:
    g_fm = np.array([1.0, 0.0])
    g_r = np.array([0.0, 1.0])
    update = conflict_free_update(g_fm, g_r)
    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(update, expected)


def test_conflict_free_update_conflicting() -> None:
    g_fm = np.array([1.0, 0.0])
    g_r = np.array([-1.0, 1.0])
    update = conflict_free_update(g_fm, g_r)
    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(update, expected)


def test_conflict_free_update_cooperative() -> None:
    g_fm = np.array([1.0, 0.0])
    g_r = np.array([1.0, 1.0])
    update = conflict_free_update(g_fm, g_r)
    u_fm = g_fm / np.linalg.norm(g_fm)
    u_r = g_r / np.linalg.norm(g_r)
    expected = (u_fm + u_r) / np.linalg.norm(u_fm + u_r)
    assert np.allclose(update, expected)


def test_conflict_free_update_shape_mismatch() -> None:
    g_fm = np.array([1.0, 0.0])
    g_r = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        conflict_free_update(g_fm, g_r)


def test_conflict_free_update_zero_gradients() -> None:
    g_fm = np.zeros(2)
    g_r = np.zeros(2)
    update = conflict_free_update(g_fm, g_r)
    assert np.allclose(update, np.zeros(2))

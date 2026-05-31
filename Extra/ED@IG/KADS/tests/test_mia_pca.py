import os
import sys
import random
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MIA import build_pca_order


def _make_images(n, directory):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(42)
    files = []
    for i in range(n):
        arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        path = os.path.join(directory, f"img_{i}.jpg")
        Image.fromarray(arr).save(path)
        files.append(path)
    return files


def test_build_pca_order_returns_subsets(tmp_path):
    """반환 리스트가 입력의 부분집합인지 확인"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m_sorted, nm_sorted = build_pca_order(m, nm)

    assert len(m_sorted) <= len(m)
    assert len(nm_sorted) <= len(nm)
    assert set(m_sorted).issubset(set(m))
    assert set(nm_sorted).issubset(set(nm))


def test_build_pca_order_is_deterministic(tmp_path):
    """같은 입력 → 같은 출력 순서"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m1, nm1 = build_pca_order(m, nm)
    m2, nm2 = build_pca_order(m, nm)

    assert m1 == m2
    assert nm1 == nm2


def test_build_pca_order_input_order_independent(tmp_path):
    """입력 순서가 달라도 출력 순서는 동일해야 한다 (nested subset 보장)"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m_sorted, _ = build_pca_order(m, nm)

    shuffled = m[:]
    random.shuffle(shuffled)
    m_sorted2, _ = build_pca_order(shuffled, nm)

    assert m_sorted == m_sorted2

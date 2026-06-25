"""E3c：select_pso_profile 规则测试。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_mock_snapshots
from src.pso_profile import get_fixed_pso_profile, select_pso_profile, snapshot_spread_pct


def _snap(record_id: str):
    return next(s for s in load_mock_snapshots() if s.snapshot_id == record_id)


def test_large_spread_profile():
    profile = select_pso_profile(_snap("snap_004"))
    assert profile["name"] == "large_spread"
    assert profile["num_particles"] == 1500
    assert profile["max_iter"] == 120


def test_low_lag_profile():
    profile = select_pso_profile(_snap("snap_008"))
    assert profile["name"] == "low_lag"
    assert profile["num_particles"] == 500
    assert profile["max_iter"] == 60


def test_default_profile():
    profile = select_pso_profile(_snap("snap_001"))
    assert profile["name"] == "default"
    assert profile["num_particles"] == 1000
    assert profile["max_iter"] == 100


def test_spread_priority_over_low_lag():
    """大价差优先于低延迟（snap_009 价差 >1% 且 lag=650）。"""
    profile = select_pso_profile(_snap("snap_009"))
    assert profile["name"] == "large_spread"


def test_fixed_profile_matches_config():
    profile = get_fixed_pso_profile()
    assert profile["name"] == "fixed"
    assert profile["num_particles"] == 1000
    assert profile["max_iter"] == 100


def test_snapshot_spread_pct():
    spread = snapshot_spread_pct(_snap("snap_004"))
    assert spread > 1.0

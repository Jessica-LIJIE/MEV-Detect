import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_mock_snapshots
from src.models import CostModel, get_search_bounds


def test_load_mock_snapshots():
    snapshots = load_mock_snapshots()
    assert len(snapshots) == 10
    assert snapshots[0].snapshot_id == "snap_001"


def test_evaluate_fitness_vectorized():
    snapshot = load_mock_snapshots()[0]
    model = CostModel()
    bounds = get_search_bounds("cpu")
    positions = torch.rand(32, 4) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    fitness = model.evaluate_fitness(positions, snapshot)
    assert fitness.shape == (32,)
    assert fitness.dtype == torch.float32


def test_snap_010_negative_spread():
    snapshots = load_mock_snapshots()
    snap_010 = next(s for s in snapshots if s.snapshot_id == "snap_010")
    model = CostModel()
    fitness = model.evaluate_fitness_scalar(10.0, 0, 0, 0, snap_010)
    assert fitness < 0

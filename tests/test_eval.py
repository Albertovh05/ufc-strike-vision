"""Tests for the evaluation harness."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.eval import StrikePrediction, evaluate, run_holdout_eval


FAKE_FIGHT_ID = "test_fight_abc123"

FAKE_GROUND_TRUTH = {
    "fight_id": FAKE_FIGHT_ID,
    "fighters": ["Fighter A", "Fighter B"],
    "rounds": [
        {
            "round": 1,
            "fighter_0_sig_str": "20 of 40",
            "fighter_1_sig_str": "15 of 30",
            "fighter_0_total_str": "30 of 50",
            "fighter_1_total_str": "20 of 35",
        },
        {
            "round": 2,
            "fighter_0_sig_str": "10 of 25",
            "fighter_1_sig_str": "18 of 32",
            "fighter_0_total_str": "15 of 30",
            "fighter_1_total_str": "22 of 40",
        },
    ],
    "url": "http://ufcstats.com/fight-details/test_fight_abc123",
}


@pytest.fixture
def holdout_fight(tmp_path, monkeypatch):
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / f"{FAKE_FIGHT_ID}.json").write_text(json.dumps(FAKE_GROUND_TRUTH))
    monkeypatch.setattr("src.eval.HOLDOUT_DIR", holdout)
    return holdout


def perfect_predictions() -> list[StrikePrediction]:
    return [
        StrikePrediction(FAKE_FIGHT_ID, 1, 0, 20, 30),
        StrikePrediction(FAKE_FIGHT_ID, 1, 1, 15, 20),
        StrikePrediction(FAKE_FIGHT_ID, 2, 0, 10, 15),
        StrikePrediction(FAKE_FIGHT_ID, 2, 1, 18, 22),
    ]


def test_perfect_predictions_zero_mae(holdout_fight):
    result = evaluate(perfect_predictions(), FAKE_FIGHT_ID)
    assert result.mae_sig == 0.0
    assert result.mae_total == 0.0


def test_off_by_one_mae(holdout_fight):
    preds = perfect_predictions()
    preds[0] = StrikePrediction(FAKE_FIGHT_ID, 1, 0, 21, 31)  # +1 on both
    result = evaluate(preds, FAKE_FIGHT_ID)
    assert result.mae_sig == pytest.approx(0.25)   # 1 error / 4 rows
    assert result.mae_total == pytest.approx(0.25)


def test_per_round_length(holdout_fight):
    result = evaluate(perfect_predictions(), FAKE_FIGHT_ID)
    assert len(result.per_round) == 4  # 2 rounds × 2 fighters


def test_run_holdout_eval_summary(holdout_fight):
    df = run_holdout_eval({FAKE_FIGHT_ID: perfect_predictions()})
    assert FAKE_FIGHT_ID in df["fight_id"].values
    assert df[df["fight_id"] == FAKE_FIGHT_ID]["mae_sig"].iloc[0] == 0.0


def test_missing_predictions_treated_as_zero(holdout_fight):
    result = evaluate([], FAKE_FIGHT_ID)
    assert result.mae_sig > 0

"""Evaluation harness — compare predicted strike counts against UFCStats ground truth."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HOLDOUT_DIR = Path(__file__).parent.parent / "data" / "holdout"


@dataclass
class StrikePrediction:
    fight_id: str
    round_number: int
    fighter_index: int  # 0 or 1
    sig_strikes: int
    total_strikes: int


@dataclass
class EvalResult:
    fight_id: str
    mae_sig: float
    mae_total: float
    per_round: list[dict] = field(default_factory=list)


def load_ground_truth(fight_id: str) -> pd.DataFrame:
    path = HOLDOUT_DIR / f"{fight_id}.json"
    data = json.loads(path.read_text())
    rows = []
    for r in data["rounds"]:
        for fi in range(2):
            sig_raw = r[f"fighter_{fi}_sig_str"].split(" of ")[0]
            tot_raw = r[f"fighter_{fi}_total_str"].split(" of ")[0]
            rows.append(
                {
                    "round": r["round"],
                    "fighter_index": fi,
                    "sig_strikes": int(sig_raw) if sig_raw.isdigit() else 0,
                    "total_strikes": int(tot_raw) if tot_raw.isdigit() else 0,
                }
            )
    return pd.DataFrame(rows)


def evaluate(predictions: list[StrikePrediction], fight_id: str) -> EvalResult:
    gt = load_ground_truth(fight_id)
    pred_df = pd.DataFrame(
        [
            {
                "round": p.round_number,
                "fighter_index": p.fighter_index,
                "pred_sig": p.sig_strikes,
                "pred_total": p.total_strikes,
            }
            for p in predictions
        ]
    )
    merged = gt.merge(pred_df, on=["round", "fighter_index"], how="left").fillna(0)
    merged["err_sig"] = (merged["sig_strikes"] - merged["pred_sig"]).abs()
    merged["err_total"] = (merged["total_strikes"] - merged["pred_total"]).abs()

    per_round = merged.to_dict(orient="records")
    return EvalResult(
        fight_id=fight_id,
        mae_sig=float(merged["err_sig"].mean()),
        mae_total=float(merged["err_total"].mean()),
        per_round=per_round,
    )


def run_holdout_eval(predictions_by_fight: dict[str, list[StrikePrediction]]) -> pd.DataFrame:
    """Evaluate all holdout fights and return a summary DataFrame."""
    results = []
    for fight_id, preds in predictions_by_fight.items():
        r = evaluate(preds, fight_id)
        results.append(
            {"fight_id": r.fight_id, "mae_sig": r.mae_sig, "mae_total": r.mae_total}
        )
    df = pd.DataFrame(results)
    if not df.empty:
        df.loc["mean"] = df[["mae_sig", "mae_total"]].mean()
    return df

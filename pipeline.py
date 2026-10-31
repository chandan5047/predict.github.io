"""Reproducible, point-in-time NBA baseline pipeline.

Run: python pipeline.py
Uses only real NBA API observations. No simulated game logs.
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, scoreboardv2
from nba_api.stats.static import teams
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "data" / "predictions.json"
SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
TARGETS = {"pts": "PTS", "reb": "REB", "fg3m": "FG3M"}
FEATURES = ["MIN_avg5", "MIN_avg10", "PTS_avg5", "PTS_avg10", "REB_avg5",
            "REB_avg10", "FG3M_avg5", "FG3M_avg10", "FGA_avg10", "FG3A_avg10",
            "FTA_avg10", "REST_DAYS", "HOME", "GAMES_PLAYED"]
TEAM_ABBR = {int(x["id"]): x["abbreviation"] for x in teams.get_teams()}

def fetch_logs(season):
    CACHE.mkdir(exist_ok=True)
    file = CACHE / ("player_games_" + season + ".csv")
    # Past completed seasons may be reused; current season refreshed every run.
    if file.exists() and season != SEASONS[-1]:
        return pd.read_csv(file)
    for attempt in range(4):
        try:
            endpoint = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation="P", timeout=75
            )
            df = endpoint.get_data_frames()[0]
            if df.empty or not {"PLAYER_ID", "GAME_ID", "GAME_DATE", "PTS", "REB", "FG3M"}.issubset(df.columns):
                raise ValueError(f"Empty or invalid player log for {season}")
            df.to_csv(file, index=False)
            time.sleep(1.5)
            return df
        except Exception as exc:
            logging.warning("Fetch %s attempt %d: %s", season, attempt + 1, exc)
            if attempt == 3:
                if file.exists():
                    logging.warning("Using cached %s observations after API failure", season)
                    return pd.read_csv(file)
                raise
            time.sleep(4 * (attempt + 1))

def historical_frame():
    frames = []
    for season in SEASONS:
        try:
            df = fetch_logs(season)
            df["SEASON"] = season
            frames.append(df)
        except Exception as exc:
            if season == SEASONS[-1] and frames:
                logging.warning("Current season unavailable: %s", exc)
            else:
                raise
    if not frames:
        raise RuntimeError("No NBA observations downloaded")
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    for col in ["MIN", "PTS", "REB", "FG3M", "FGA", "FG3A", "FTA"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["GAME_DATE", "PLAYER_ID", "GAME_ID", "PTS"]).copy()
    df = df.drop_duplicates(["PLAYER_ID", "GAME_ID"]).sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    match = df["MATCHUP"].astype(str)
    df["HOME"] = (~match.str.contains("@", regex=False)).astype(int)
    df["OPPONENT"] = match.str.extract(r"(?:vs\\.|@)\\s*([A-Z]{3})", expand=False)
    g = df.groupby("PLAYER_ID", sort=False)
    for stat in ["MIN", "PTS", "REB", "FG3M", "FGA", "FG3A", "FTA"]:
        for window in ([5, 10] if stat in ["MIN", "PTS", "REB", "FG3M"] else [10]):
            df[f"{stat}_avg{window}"] = g[stat].transform(
                lambda s: s.shift(1).rolling(window, min_periods=3).mean()
            )
    df["REST_DAYS"] = (g["GAME_DATE"].diff().dt.days - 1).clip(lower=0, upper=10)
    df["GAMES_PLAYED"] = g.cumcount()
    return df

def fit_models(df):
    models, diagnostics, residual_sds = {}, {}, {}
    trainable = df[df["GAMES_PLAYED"] >= 5].sort_values("GAME_DATE").copy()
    if len(trainable) < 300:
        raise RuntimeError("Insufficient real observations to train reliable baselines")
    dates = sorted(trainable.GAME_DATE.unique())
    cutoff = dates[int(len(dates) * .8)]
    train, test = trainable[trainable.GAME_DATE < cutoff], trainable[trainable.GAME_DATE >= cutoff]
    if train.empty or test.empty:
        raise RuntimeError("Chronological validation split is empty")
    for alias, target in TARGETS.items():
        tr = train.dropna(subset=[target])
        te = test.dropna(subset=[target])
        model = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(), Ridge(alpha=100.))
        model.fit(tr[FEATURES], tr[target])
        in_pred = np.clip(model.predict(tr[FEATURES]), 0, None)
        pred = np.clip(model.predict(te[FEATURES]), 0, None)
        in_base = tr[target + "_avg5"].fillna(tr[target].mean()).clip(lower=0)
        base = te[target + "_avg5"].fillna(tr[target].mean()).clip(lower=0)
        err = te[target].to_numpy() - pred
        residual_sds[alias] = max(float(np.std(err, ddof=1)), .2)
        def scores(actual, estimates, baseline):
            return {
                "n": int(len(actual)),
                "mae": round(float(mean_absolute_error(actual, estimates)), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(actual, estimates))), 3),
                "bias": round(float(np.mean(estimates - np.asarray(actual))), 3),
                "r2": round(float(r2_score(actual, estimates)), 3),
                "baseline_mae": round(float(mean_absolute_error(actual, baseline)), 3),
            }
        diagnostics[alias] = {
            "in_sample": scores(tr[target], in_pred, in_base),
            "out_of_sample": scores(te[target], pred, base),
            "residual_sd": round(residual_sds[alias], 3),
            "train_through": str(pd.Timestamp(tr.GAME_DATE.max()).date()),
            "test_from": str(pd.Timestamp(te.GAME_DATE.min()).date()),
        }
        # Refit only on data available before the day of prediction.
        full = trainable.dropna(subset=[target])
        model.fit(full[FEATURES], full[target])
        models[alias] = model
    return models, diagnostics, residual_sds

def upcoming_games():
    today = datetime.now(timezone.utc).strftime("%m/%d/%Y")
    for attempt in range(3):
        try:
            score = scoreboardv2.ScoreboardV2(game_date=today, timeout=60)
            games = score.game_header.get_data_frame()
            return games
        except Exception as exc:
            logging.warning("Scoreboard attempt %s failed: %s", attempt + 1, exc)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Live scoreboard is not accessible; refusing to invent matchups")

def forecast():
    df = historical_frame()
    models, diagnostics, sd = fit_models(df)
    schedule = upcoming_games()
    if schedule.empty:
        logging.info("No games on today's scoreboard; publishing empty prediction list")
    latest = df.sort_values("GAME_DATE").groupby("PLAYER_ID", as_index=False).tail(1)
    # Only recent historical team members are eligible; no injury verification.
    recent = df.GAME_DATE.max() - pd.Timedelta(days=28)
    latest = latest[latest.GAME_DATE >= recent]
    output = []
    for _, game in schedule.iterrows():
        home_id, away_id = int(game["HOME_TEAM_ID"]), int(game["VISITOR_TEAM_ID"])
        for team_id, opp_id, home in [(home_id, away_id, 1), (away_id, home_id, 0)]:
            players = latest[latest.TEAM_ID.astype(int) == team_id].copy()
            for _, r in players.iterrows():
                x = r[FEATURES].copy()
                x["HOME"] = home
                x["REST_DAYS"] = min(max((pd.Timestamp.now() - r.GAME_DATE).days - 1, 0), 10)
                minutes = max(float(r["MIN_avg10"]) if pd.notna(r["MIN_avg10"]) else 0., 0.)
                if minutes < 5 or r["GAMES_PLAYED"] < 8:
                    continue
                estimates = {}
                for name, model in models.items():
                    value = float(np.clip(model.predict(pd.DataFrame([x])[FEATURES])[0], 0, None))
                    estimates[name] = round(value, 2)
                    estimates[name + "_sd"] = round(sd[name], 2)
                output.append({
                    "game_date": pd.Timestamp(game["GAME_DATE_EST"]).strftime("%Y-%m-%d"),
                    "game_key": str(game["GAME_ID"]),
                    "matchup": f"{TEAM_ABBR.get(away_id,away_id)} @ {TEAM_ABBR.get(home_id,home_id)}",
                    "player_id": int(r.PLAYER_ID), "player": str(r.PLAYER_NAME),
                    "team": TEAM_ABBR.get(team_id, str(team_id)),
                    "opponent": TEAM_ABBR.get(opp_id, str(opp_id)),
                    "minutes": round(minutes, 1),
                    "warning": "Availability, injuries and starting role unverified",
                    **estimates
                })
    OUT.parent.mkdir(exist_ok=True)
    summary = "; ".join(f"{k.upper()} in-sample MAE {v['in_sample']['mae']}, out-of-sample MAE {v['out_of_sample']['mae']} (baseline {v['out_of_sample']['baseline_mae']})"
                        for k, v in diagnostics.items())
    payload = {"metadata": {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_summary": summary,
        "scope": "Today only; season-to-date NBA logs; injuries not integrated",
        "models": diagnostics,
    }, "predictions": output}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT)
    logging.info("Published %d forecasts to %s", len(output), OUT)

if __name__ == "__main__":
    forecast()

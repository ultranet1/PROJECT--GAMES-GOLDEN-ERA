"""
generate_dummy_data.py
=======================
Generates synthetic ("dummy") CSV data for the "When Was the Golden Era of
Video Games?" project, matching the schema described in the README:

    - game_sales.csv             (raw table)
    - reviews.csv                (raw table)
    - users_avg_year_rating.csv  (derived table, computed from the two above)
    - critics_avg_year_rating.csv(derived table, computed from the two above)

This data is entirely made up for demonstration purposes - it is NOT the
real Kaggle "Video Game Sales with Ratings" dataset. Run this script to
regenerate the CSVs in data/ with a fresh random sample.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_GAMES = 400
OUT_DIR = Path(__file__).resolve().parent / "data"

random.seed(SEED)
np.random.seed(SEED)

PLATFORMS = ["PC", "PS4", "PS3", "PS2", "PS1", "Xbox One", "Xbox 360",
             "Xbox", "Wii", "Wii U", "Switch", "NES", "SNES", "N64",
             "GameCube", "3DS", "DS", "GBA"]

PUBLISHERS = ["Nintendo", "Sony Interactive", "Microsoft Studios", "Electronic Arts",
              "Activision", "Ubisoft", "Take-Two Interactive", "Square Enix",
              "Capcom", "Sega", "Bandai Namco", "Bethesda Softworks",
              "Valve", "CD Projekt", "Konami", "Epic Games"]

DEVELOPERS = ["Nintendo EAD", "Naughty Dog", "343 Industries", "DICE",
              "Infinity Ward", "Ubisoft Montreal", "Rockstar North",
              "Square Enix Creative", "Capcom R&D", "Sonic Team",
              "From Software", "id Software", "Valve", "CD Projekt Red",
              "Kojima Productions", "Insomniac Games", "Bungie", "Retro Studios"]

ADJECTIVES = ["Legend", "Shadow", "Eternal", "Rogue", "Crimson", "Silent",
              "Ancient", "Iron", "Golden", "Dark", "Lost", "Final", "Wild",
              "Broken", "Hidden", "Astral", "Neon", "Frozen"]

NOUNS = ["Kingdom", "Odyssey", "Warfare", "Chronicles", "Horizon", "Empire",
         "Legacy", "Frontier", "Rebellion", "Saga", "Dominion", "Nexus",
         "Uprising", "Requiem", "Vanguard", "Genesis", "Wasteland", "Arena"]

SUBTITLES = ["", " II", " III", " Remastered", ": Reborn", ": Origins",
             ": Redux", " 2", " 4", ": The Final Chapter", ": New Horizons"]

used_names: set[str] = set()


def make_game_name() -> str:
    """Produce a plausible-looking, unique fake game title."""
    while True:
        name = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}{random.choice(SUBTITLES)}"
        if name not in used_names:
            used_names.add(name)
            return name


def make_sales_figure(year: int) -> float:
    """
    Most games in a "top 400" list sell a modest amount, with a handful of
    blockbusters. Model this with a heavy-tailed lognormal distribution,
    scaled down slightly for older years (smaller install base).
    """
    base = np.random.lognormal(mean=0.4, sigma=0.9)
    era_factor = 0.6 if year < 1995 else 1.0
    return round(min(base * era_factor, 85.0), 2)


def build_game_sales(n: int) -> pd.DataFrame:
    years = np.random.choice(range(1977, 2021), size=n,
                              p=_year_weights())
    rows = []
    for year in years:
        rows.append({
            "name": make_game_name(),
            "platform": random.choice(PLATFORMS),
            "publisher": random.choice(PUBLISHERS),
            "developer": random.choice(DEVELOPERS),
            "games_sold": make_sales_figure(int(year)),
            "year": int(year),
        })
    df = pd.DataFrame(rows)
    # Give each game a "for <platform>" style full title, matching the
    # original dataset's naming convention seen in the notebook output
    # (e.g. "Wii Sports for Wii").
    df["name"] = df["name"] + " for " + df["platform"]
    return df


def _year_weights() -> np.ndarray:
    """Weight more games toward the 1995-2018 boom years, fewer at the tails."""
    years = np.arange(1977, 2021)
    center = 2007
    weights = np.exp(-0.5 * ((years - center) / 12) ** 2)
    return weights / weights.sum()


# A handful of years get a quality boost so the data actually tells a
# "golden era" story once aggregated - otherwise every year regresses to
# the same mean and the golden_years query returns nothing interesting.
GOLDEN_YEARS = {1998: 1.9, 2004: 1.7, 2009: 1.8, 2010: 1.6}


def build_reviews(game_sales: pd.DataFrame, drop_fraction: float = 0.05) -> pd.DataFrame:
    """
    Build a reviews table keyed on the same `name`. A small fraction of
    games are deliberately left out, mirroring the real-world case where
    not every game in the sales table has a matching review (so the
    INNER JOIN in the queries doesn't just return everything 1:1).
    """
    year_by_name = dict(zip(game_sales["name"], game_sales["year"]))
    names = game_sales["name"].tolist()
    n_drop = int(len(names) * drop_fraction)
    reviewed_names = names.copy()
    random.shuffle(reviewed_names)
    reviewed_names = reviewed_names[: len(names) - n_drop]

    rows = []
    for name in reviewed_names:
        boost = GOLDEN_YEARS.get(int(year_by_name[name]), 0.0)
        # Critic and user scores are correlated but noisy, with critics
        # trending slightly higher on average and users occasionally
        # loving (or hating) a game critics were lukewarm on.
        critic = np.clip(np.random.normal(loc=7.2 + boost, scale=0.9), 0, 10)
        user = np.clip(critic + np.random.normal(loc=-0.1, scale=0.7), 0, 10)
        rows.append({
            "name": name,
            "critic_score": round(float(critic), 1),
            "user_score": round(float(user), 1),
        })
    return pd.DataFrame(rows)


def build_year_ratings(game_sales: pd.DataFrame, reviews: pd.DataFrame):
    """Recreate the two derived tables using the same logic as the SQL queries."""
    merged = game_sales.merge(reviews, on="name", how="inner")

    critics = (
        merged.groupby("year")
        .agg(num_games=("name", "count"), avg_critic_score=("critic_score", "mean"))
        .reset_index()
    )
    critics["avg_critic_score"] = critics["avg_critic_score"].round(2)

    users = (
        merged.groupby("year")
        .agg(num_games=("name", "count"), avg_user_score=("user_score", "mean"))
        .reset_index()
    )
    users["avg_user_score"] = users["avg_user_score"].round(2)

    return users, critics


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    game_sales = build_game_sales(N_GAMES)
    reviews = build_reviews(game_sales)
    users_avg_year_rating, critics_avg_year_rating = build_year_ratings(game_sales, reviews)

    game_sales.to_csv(OUT_DIR / "game_sales.csv", index=False)
    reviews.to_csv(OUT_DIR / "reviews.csv", index=False)
    users_avg_year_rating.to_csv(OUT_DIR / "users_avg_year_rating.csv", index=False)
    critics_avg_year_rating.to_csv(OUT_DIR / "critics_avg_year_rating.csv", index=False)

    print(f"Wrote {len(game_sales)} rows to game_sales.csv")
    print(f"Wrote {len(reviews)} rows to reviews.csv")
    print(f"Wrote {len(users_avg_year_rating)} rows to users_avg_year_rating.csv")
    print(f"Wrote {len(critics_avg_year_rating)} rows to critics_avg_year_rating.csv")


if __name__ == "__main__":
    main()

"""
analysis.py
============
Pandas equivalents of the three SQL queries in the README, run directly
against the CSVs in data/. Useful if you'd rather not spin up a database
just to explore the project.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_data():
    game_sales = pd.read_csv(DATA_DIR / "game_sales.csv")
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    users_avg_year_rating = pd.read_csv(DATA_DIR / "users_avg_year_rating.csv")
    critics_avg_year_rating = pd.read_csv(DATA_DIR / "critics_avg_year_rating.csv")
    return game_sales, reviews, users_avg_year_rating, critics_avg_year_rating


def best_selling_games(game_sales: pd.DataFrame) -> pd.DataFrame:
    """SELECT * FROM game_sales ORDER BY games_sold DESC LIMIT 10;"""
    return game_sales.sort_values("games_sold", ascending=False).head(10)


def critics_top_ten_years(game_sales: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    SELECT g.year, COUNT(g.name) AS num_games, ROUND(AVG(r.critic_score), 2) AS avg_critic_score
    FROM game_sales g INNER JOIN reviews r ON g.name = r.name
    GROUP BY g.year HAVING COUNT(g.name) >= 4
    ORDER BY avg_critic_score DESC LIMIT 10;
    """
    merged = game_sales.merge(reviews, on="name", how="inner")
    grouped = (
        merged.groupby("year")
        .agg(num_games=("name", "count"), avg_critic_score=("critic_score", "mean"))
        .reset_index()
    )
    grouped["avg_critic_score"] = grouped["avg_critic_score"].round(2)
    grouped = grouped[grouped["num_games"] >= 4]
    return grouped.sort_values("avg_critic_score", ascending=False).head(10)


def golden_years(users: pd.DataFrame, critics: pd.DataFrame) -> pd.DataFrame:
    """
    SELECT u.year, u.num_games, c.avg_critic_score, u.avg_user_score,
           c.avg_critic_score - u.avg_user_score AS diff
    FROM critics_avg_year_rating c INNER JOIN users_avg_year_rating u ON c.year = u.year
    WHERE c.avg_critic_score > 9 OR u.avg_user_score > 9
    ORDER BY c.year ASC;
    """
    merged = critics.merge(users, on="year", suffixes=("_critic", "_user"))
    merged["diff"] = merged["avg_critic_score"] - merged["avg_user_score"]
    filtered = merged[(merged["avg_critic_score"] > 9) | (merged["avg_user_score"] > 9)]
    cols = ["year", "num_games_user", "avg_critic_score", "avg_user_score", "diff"]
    return filtered.sort_values("year")[cols]


def main() -> None:
    game_sales, reviews, users, critics = load_data()

    print("=== Best-selling games ===")
    print(best_selling_games(game_sales).to_string(index=False))
    print()

    print("=== Critics' top ten years ===")
    print(critics_top_ten_years(game_sales, reviews).to_string(index=False))
    print()

    print("=== Golden years of gaming ===")
    gy = golden_years(users, critics)
    if gy.empty:
        print("No years crossed the >9 threshold in this sample - try regenerating the data.")
    else:
        print(gy.to_string(index=False))


if __name__ == "__main__":
    main()

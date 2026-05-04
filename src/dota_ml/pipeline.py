from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from scipy.sparse import csr_array, csr_matrix, hstack
from sklearn.linear_model import LogisticRegression


@dataclass
class RegionMmrArtifacts:
    target_encoder: TargetEncoder
    mmr_median: float


@dataclass
class BasePipelineResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    player_df_clean: pd.DataFrame
    X_all_train: csr_matrix
    X_all_test: csr_matrix
    model: LogisticRegression
    submission: pd.DataFrame
    region_mmr_artifacts: RegionMmrArtifacts
    hero_encoder: "HeroesEncoder"
    submission_path: str


def load_base_data(
    data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "matches_df_train.csv")
    test_df = pd.read_csv(data_dir / "matches_df_test.csv")
    player_df = pd.read_csv(data_dir / "player_df.csv")
    return train_df, test_df, player_df


def clean_player_df(
    player_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    suspicious_id: int = 4294967295,
) -> pd.DataFrame:
    df = player_df.copy()

    all_matches = pd.Index(train_df["match_id"]).union(test_df["match_id"])
    df = df[df["match_id"].isin(all_matches)].copy()

    df = df.dropna(subset=["match_id", "account_id", "hero_id", "player_slot"]).copy()

    hero0_matches = df.loc[df["hero_id"].eq(0), "match_id"].unique()
    df = df[~df["match_id"].isin(hero0_matches)].copy()

    is_radiant = df["player_slot"].between(0, 4)
    is_dire = df["player_slot"].between(128, 132)
    df = df[is_radiant | is_dire].copy()

    df["side"] = np.where(df["player_slot"].between(0, 4), "radiant", "dire")

    pairs_sides = (
        df[df["account_id"] != suspicious_id]
        .groupby(["match_id", "account_id"])["side"]
        .nunique()
    )
    bad_pairs = pairs_sides[pairs_sides > 1].index
    pair_index = pd.MultiIndex.from_frame(df[["match_id", "account_id"]])
    df = df[~pair_index.isin(bad_pairs)].copy()

    cnt = df.groupby(["match_id", "side"]).size().unstack(fill_value=0)
    good_matches = cnt.index[(cnt.get("radiant", 0) == 5) & (cnt.get("dire", 0) == 5)]
    df = df[df["match_id"].isin(good_matches)].copy()

    final_cnt = df.groupby(["match_id", "side"]).size().unstack(fill_value=0)
    if not ((final_cnt["radiant"] == 5).all() and (final_cnt["dire"] == 5).all()):
        raise ValueError("Not all matches are strict 5v5 after cleaning.")

    return df.drop(columns=["side"]).copy()


class HeroesEncoder:
    def __init__(
        self,
        match_col: str = "match_id",
        hero_col: str = "hero_id",
        slot_col: str = "player_slot",
    ):
        self.match_col = match_col
        self.hero_col = hero_col
        self.slot_col = slot_col
        self.hero_to_col_: dict[int, int] | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HeroesEncoder":
        heroes = (
            pd.Series(X[self.hero_col].dropna().astype(int).unique())
            .sort_values()
            .tolist()
        )
        self.hero_to_col_ = {h: i for i, h in enumerate(heroes)}
        self.feature_names_ = [f"hero_{h}" for h in heroes]
        return self

    def transform(self, X: pd.DataFrame, y: Iterable[int] | None = None) -> csr_array:
        if self.hero_to_col_ is None:
            raise ValueError("Encoder is not fitted. Call fit() first.")

        if y is None:
            match_ids = pd.Index(X[self.match_col].drop_duplicates())
        else:
            match_ids = pd.Index(y)

        match_to_row = {m: i for i, m in enumerate(match_ids)}

        work = X[[self.match_col, self.hero_col, self.slot_col]].copy()
        slots = work[self.slot_col].to_numpy()

        side = np.zeros(len(work), dtype=np.int8)
        side[(slots >= 0) & (slots <= 4)] = 1
        side[(slots >= 128) & (slots <= 132)] = -1

        rows = work[self.match_col].map(match_to_row).to_numpy()
        cols = work[self.hero_col].map(self.hero_to_col_).to_numpy()

        valid = (side != 0) & (~pd.isna(rows)) & (~pd.isna(cols))

        rows = rows[valid].astype(np.int32)
        cols = cols[valid].astype(np.int32)
        data = side[valid].astype(np.int8)

        mat = csr_array(
            (data, (rows, cols)),
            shape=(len(match_ids), len(self.hero_to_col_)),
            dtype=np.int8,
        )
        mat.sum_duplicates()
        mat.data = np.sign(mat.data).astype(np.int8)
        return mat

    def get_feature_names_out(self) -> np.ndarray:
        if self.feature_names_ is None:
            raise ValueError("Encoder is not fitted.")
        return np.array(self.feature_names_, dtype=object)


def fit_region_mmr_artifacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    region_col: str = "region",
    mmr_col: str = "avg_mmr",
) -> RegionMmrArtifacts:
    te = TargetEncoder(cols=[region_col])
    te.fit(X_train[[region_col]], y_train)
    mmr_median = float(X_train[mmr_col].median())
    return RegionMmrArtifacts(target_encoder=te, mmr_median=mmr_median)


def transform_region_mmr(
    X: pd.DataFrame,
    artifacts: RegionMmrArtifacts,
    region_col: str = "region",
    mmr_col: str = "avg_mmr",
) -> csr_matrix:
    region_te = artifacts.target_encoder.transform(X[[region_col]])[region_col]
    mmr_missing = X[mmr_col].isna().astype(int)
    mmr_feat = np.sqrt(X[mmr_col].fillna(artifacts.mmr_median))
    return csr_matrix(np.c_[region_te, mmr_missing, mmr_feat])


def build_hero_matrices(
    player_df_clean: pd.DataFrame,
    train_match_ids: Iterable[int],
    test_match_ids: Iterable[int],
) -> tuple[HeroesEncoder, csr_array, csr_array]:
    hero_encoder = HeroesEncoder().fit(player_df_clean)

    train_match_ids = pd.Index(train_match_ids)
    test_match_ids = pd.Index(test_match_ids)

    player_train = player_df_clean[player_df_clean["match_id"].isin(train_match_ids)]
    player_test = player_df_clean[player_df_clean["match_id"].isin(test_match_ids)]

    X_hero_train = hero_encoder.transform(player_train, y=train_match_ids)
    X_hero_test = hero_encoder.transform(player_test, y=test_match_ids)
    return hero_encoder, X_hero_train, X_hero_test


def build_all_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    player_df_clean: pd.DataFrame,
) -> tuple[csr_matrix, csr_matrix, RegionMmrArtifacts, HeroesEncoder]:
    artifacts = fit_region_mmr_artifacts(X_train, y_train)
    X_region_mmr_train = transform_region_mmr(X_train, artifacts)
    X_region_mmr_test = transform_region_mmr(X_test, artifacts)

    hero_encoder, X_hero_train, X_hero_test = build_hero_matrices(
        player_df_clean,
        train_match_ids=X_train["match_id"],
        test_match_ids=X_test["match_id"],
    )

    X_all_train = hstack([X_region_mmr_train, X_hero_train], format="csr")
    X_all_test = hstack([X_region_mmr_test, X_hero_test], format="csr")
    return X_all_train, X_all_test, artifacts, hero_encoder


def make_submission(
    match_ids: Iterable[int],
    pred_proba: np.ndarray,
    id_col: str = "ID",
    value_col: str = "Value",
) -> pd.DataFrame:
    return pd.DataFrame(
        {id_col: np.asarray(match_ids), value_col: np.asarray(pred_proba)}
    )


def run_base_pipeline(
    data_dir: str | Path,
    model_params: dict | None = None,
    submission_path: str | Path = "submission_base_all_features.csv",
) -> BasePipelineResult:
    train_df, test_df, player_df = load_base_data(data_dir)

    X_train = train_df.drop(columns=["radiant_win"]).copy()
    y_train = train_df["radiant_win"].astype(int).copy()
    X_test = test_df.copy()

    player_df_clean = clean_player_df(player_df, train_df, test_df)

    X_all_train, X_all_test, region_mmr_artifacts, hero_encoder = build_all_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        player_df_clean=player_df_clean,
    )

    final_params = {"random_state": 42}
    if model_params is not None:
        final_params.update(model_params)

    model = LogisticRegression(**final_params)
    model.fit(X_all_train, y_train)

    test_pred = model.predict_proba(X_all_test)[:, 1]
    submission = make_submission(
        match_ids=X_test["match_id"].values, pred_proba=test_pred
    )

    submission_path = Path(submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    return BasePipelineResult(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        player_df_clean=player_df_clean,
        X_all_train=X_all_train,
        X_all_test=X_all_test,
        model=model,
        submission=submission,
        region_mmr_artifacts=region_mmr_artifacts,
        hero_encoder=hero_encoder,
        submission_path=str(submission_path),
    )

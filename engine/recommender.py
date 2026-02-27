"""
Hybrid Anime Recommender Engine
================================
感情ベクトルと人気度を組み合わせたハイブリッド推薦エンジン。

スコア計算式:
  Final Score = 0.8 × cosine_similarity(user_vec, anime_vec)
              + 0.2 × normalized_popularity

使用方法（ライブラリとして）:
  from engine.recommender import AnimeRecommender

  rec = AnimeRecommender()
  results = rec.get_recommendations(
      user_vector={"tear_jerker": 0.9, "hype": 0.1, "healing": 0.3, "dark": 0.7, "comedy": 0.0},
      top_k=10,
  )

使用方法（スタンドアロン）:
  python engine/recommender.py [--data data/anime_with_sentiments.json]
                               [--top-k 10]
                               [--min-score 60]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# 定数
# ─────────────────────────────────────────

DEFAULT_DATA_PATH = Path("data/anime_with_sentiments.json")

# 感情次元の定義順（ベクトル化時の軸の並び）
EMOTION_KEYS: list[str] = ["tear_jerker", "hype", "healing", "dark", "comedy"]

# 感情ラベルの日本語表示
EMOTION_LABELS: dict[str, str] = {
    "tear_jerker": "涙活 (Tear-jerker)",
    "hype":        "アドレナリン (Hype)",
    "healing":     "癒やし (Healing)",
    "dark":        "衝撃/鬱 (Dark)",
    "comedy":      "爆笑 (Comedy)",
}

# ハイブリッドスコアの重み
WEIGHT_EMOTION     = 0.8
WEIGHT_POPULARITY  = 0.2


# ─────────────────────────────────────────
# データ型
# ─────────────────────────────────────────

@dataclass
class Recommendation:
    rank:              int
    anime_id:          int
    title:             str
    final_score:       float
    cosine_sim:        float
    popularity_score:  float   # 正規化済み 0.0〜1.0
    reasoning:         str
    genres:            list[str]    = field(default_factory=list)
    average_score:     float | None = None
    popularity_raw:    int   | None = None

    def as_dict(self) -> dict:
        return {
            "rank":             self.rank,
            "anime_id":         self.anime_id,
            "title":            self.title,
            "final_score":      round(self.final_score, 4),
            "cosine_sim":       round(self.cosine_sim, 4),
            "popularity_score": round(self.popularity_score, 4),
            "reasoning":        self.reasoning,
            "genres":           self.genres,
            "average_score":    self.average_score,
            "popularity_raw":   self.popularity_raw,
        }


# ─────────────────────────────────────────
# 推薦エンジン
# ─────────────────────────────────────────

class AnimeRecommender:
    """
    ハイブリッド推薦エンジン。

    Parameters
    ----------
    data_path : Path | str
        anime_with_sentiments.json のパス。
    min_avg_score : float | None
        この値未満の averageScore を持つ作品をロード時に除外する。
        None の場合はフィルタリングしない。
    """

    def __init__(
        self,
        data_path: Path | str = DEFAULT_DATA_PATH,
        min_avg_score: float | None = None,
    ):
        self._records:          list[dict]       = []
        self._sentiment_matrix: np.ndarray       = np.empty((0, 5))
        self._popularity_norm:  np.ndarray       = np.empty((0,))

        self._load(Path(data_path), min_avg_score)

    # ── ロード ──────────────────────────────

    def _load(self, path: Path, min_avg_score: float | None) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"データファイルが見つかりません: {path}\n"
                "先に sentiment_analyzer.py を実行してください。"
            )

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        anime_list: list[dict] = raw.get("anime", [])
        valid_records: list[dict] = []

        for anime in anime_list:
            sentiment = anime.get("sentiment") or {}

            # 感情分析がスキップされた作品は除外
            if sentiment.get("status") != "ok":
                continue

            # 5次元の感情スコアがすべて存在するか確認
            vec = [sentiment.get(k) for k in EMOTION_KEYS]
            if any(v is None for v in vec):
                continue

            # averageScore フィルタ
            avg = anime.get("average_score")
            if min_avg_score is not None and (avg is None or avg < min_avg_score):
                continue

            valid_records.append(anime)

        if not valid_records:
            raise ValueError(
                "有効な感情分析済みアニメが0件です。"
                " sentiment_analyzer.py の出力を確認してください。"
            )

        self._records = valid_records

        # 感情行列 (N × 5)
        self._sentiment_matrix = np.array(
            [
                [anime["sentiment"][k] for k in EMOTION_KEYS]
                for anime in valid_records
            ],
            dtype=np.float64,
        )

        # 人気度の min-max 正規化
        raw_pop = np.array(
            [anime.get("popularity") or 0 for anime in valid_records],
            dtype=np.float64,
        )
        pop_min, pop_max = raw_pop.min(), raw_pop.max()
        if pop_max > pop_min:
            self._popularity_norm = (raw_pop - pop_min) / (pop_max - pop_min)
        else:
            self._popularity_norm = np.full(len(valid_records), 0.5)

    # ── パブリック API ──────────────────────

    def get_recommendations(
        self,
        user_vector: dict[str, float] | list[float] | np.ndarray,
        top_k: int = 10,
        min_avg_score: float | None = None,
    ) -> list[Recommendation]:
        """
        ユーザーの感情ベクトルに基づいてアニメを推薦する。

        Parameters
        ----------
        user_vector : dict | list | ndarray
            5次元の感情ベクトル。
            dict の場合: {"tear_jerker": 0.9, "hype": 0.1, ...}
            list / ndarray の場合: [tear_jerker, hype, healing, dark, comedy] の順
        top_k : int
            返す推薦件数。
        min_avg_score : float | None
            この値未満の averageScore を持つ作品をクエリ時に除外する。
            ロード時フィルタと組み合わせて使える。

        Returns
        -------
        list[Recommendation]
            スコア降順の推薦リスト。
        """
        user_vec = _parse_user_vector(user_vector)

        # ゼロベクトルは類似度計算が不定になるため警告
        if np.linalg.norm(user_vec) == 0:
            raise ValueError(
                "ユーザーベクトルがゼロベクトルです。"
                " 少なくとも1つの感情スコアに 0 より大きい値を設定してください。"
            )

        # ── コサイン類似度 (1 × N) ──
        cos_sims = cosine_similarity(
            user_vec.reshape(1, -1), self._sentiment_matrix
        ).flatten()

        # ── ハイブリッドスコア ──
        final_scores = WEIGHT_EMOTION * cos_sims + WEIGHT_POPULARITY * self._popularity_norm

        # クエリ時の averageScore フィルタ（マスク適用）
        if min_avg_score is not None:
            for i, anime in enumerate(self._records):
                avg = anime.get("average_score")
                if avg is None or avg < min_avg_score:
                    final_scores[i] = -1.0   # ランク外に落とす

        # 降順ソートして上位 top_k を返す
        ranked_indices = np.argsort(final_scores)[::-1]

        results: list[Recommendation] = []
        rank = 1
        for idx in ranked_indices:
            if final_scores[idx] < 0:
                continue
            if rank > top_k:
                break

            anime = self._records[idx]
            sentiment = anime["sentiment"]
            title = (
                anime["title"].get("english")
                or anime["title"].get("romaji")
                or str(anime["id"])
            )

            results.append(
                Recommendation(
                    rank=rank,
                    anime_id=anime["id"],
                    title=title,
                    final_score=float(final_scores[idx]),
                    cosine_sim=float(cos_sims[idx]),
                    popularity_score=float(self._popularity_norm[idx]),
                    reasoning=sentiment.get("reasoning") or "",
                    genres=anime.get("genres") or [],
                    average_score=anime.get("average_score"),
                    popularity_raw=anime.get("popularity"),
                )
            )
            rank += 1

        return results

    @property
    def loaded_count(self) -> int:
        return len(self._records)


# ─────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────

def _parse_user_vector(
    user_vector: dict[str, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """入力を numpy 配列 (shape: 5,) に変換する。"""
    if isinstance(user_vector, dict):
        missing = [k for k in EMOTION_KEYS if k not in user_vector]
        if missing:
            raise KeyError(f"感情キーが不足しています: {missing}")
        vec = np.array([float(user_vector[k]) for k in EMOTION_KEYS], dtype=np.float64)
    else:
        vec = np.asarray(user_vector, dtype=np.float64)
        if vec.shape != (5,):
            raise ValueError(
                f"ユーザーベクトルは長さ5のリストか dict を渡してください (受け取ったshape: {vec.shape})"
            )
    return vec


# ─────────────────────────────────────────
# ターミナル表示
# ─────────────────────────────────────────

_BAR_MAX_LEN = 20


def _score_bar(value: float) -> str:
    """0.0〜1.0 の値を ASCII バーに変換する。"""
    filled = round(value * _BAR_MAX_LEN)
    return "█" * filled + "░" * (_BAR_MAX_LEN - filled)


def _print_recommendations(
    results: list[Recommendation],
    user_vec: dict[str, float],
) -> None:
    """推薦結果をターミナルに整形表示する。"""
    width = 72
    line = "─" * width

    print(f"\n{'='*width}")
    print(f"  推薦結果 TOP {len(results)}")
    print(f"{'='*width}")

    # 入力ベクトルの表示
    print("\n  [入力した感情ベクトル]")
    for key, label in EMOTION_LABELS.items():
        val = user_vec[key]
        print(f"    {label:<30} {_score_bar(val)}  {val:.2f}")
    print()

    for rec in results:
        print(line)
        # タイトルとランク
        print(f"  #{rec.rank:<3} {rec.title}")
        if rec.genres:
            print(f"       ジャンル: {', '.join(rec.genres[:5])}")
        if rec.average_score is not None:
            print(f"       評価スコア: {rec.average_score}/100  |  人気度: {rec.popularity_raw:,}" if rec.popularity_raw else f"       評価スコア: {rec.average_score}/100")

        # スコア
        print(f"\n       Final Score     : {rec.final_score:.4f}  {_score_bar(rec.final_score)}")
        print(f"       Cosine Sim (×0.8): {rec.cosine_sim:.4f}  {_score_bar(rec.cosine_sim)}")
        print(f"       Popularity (×0.2): {rec.popularity_score:.4f}  {_score_bar(rec.popularity_score)}")

        # 推薦理由
        if rec.reasoning:
            print(f"\n       推薦理由:")
            # 長文を折り返して表示
            words = rec.reasoning.split()
            line_buf, line_len = [], 0
            for word in words:
                if line_len + len(word) + 1 > 60:
                    print(f"         {'  '.join(line_buf)}")
                    line_buf, line_len = [word], len(word)
                else:
                    line_buf.append(word)
                    line_len += len(word) + 1
            if line_buf:
                print(f"         {' '.join(line_buf)}")

        print()

    print(f"{'='*width}\n")


def _prompt_user_vector() -> dict[str, float]:
    """ターミナルからユーザーの感情スコアをインタラクティブに入力させる。"""
    print("\n" + "="*60)
    print("  アニメ推薦エンジン - 感情ベクトル入力")
    print("="*60)
    print("  各感情の強さを 0.0〜1.0 で入力してください。")
    print("  例) 全然感じない→0.0, 少し→0.3, 強く→0.7, 最高→1.0\n")

    user_vec: dict[str, float] = {}
    for key, label in EMOTION_LABELS.items():
        while True:
            try:
                raw = input(f"  {label:<30} > ").strip()
                val = float(raw)
                if not (0.0 <= val <= 1.0):
                    raise ValueError
                user_vec[key] = val
                break
            except ValueError:
                print("    ※ 0.0〜1.0 の数値を入力してください。")

    return user_vec


# ─────────────────────────────────────────
# エントリーポイント（スタンドアロン実行）
# ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="感情ベクトルを入力してアニメを推薦するエンジン"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"感情分析済みJSONファイルのパス (デフォルト: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="推薦件数 (デフォルト: 10)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="SCORE",
        help="この値未満の averageScore を持つ作品を除外 (例: 70)",
    )
    args = parser.parse_args()

    # エンジン初期化
    try:
        recommender = AnimeRecommender(
            data_path=args.data,
            min_avg_score=args.min_score,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nエラー: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  データロード完了: {recommender.loaded_count}件の作品")

    # 感情ベクトルの入力
    user_vec = _prompt_user_vector()

    # 推薦実行
    try:
        results = recommender.get_recommendations(
            user_vector=user_vec,
            top_k=args.top_k,
            min_avg_score=args.min_score,
        )
    except ValueError as exc:
        print(f"\nエラー: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("\n推薦できる作品が見つかりませんでした。フィルタ条件を緩めてみてください。")
        sys.exit(0)

    _print_recommendations(results, user_vec)


if __name__ == "__main__":
    main()

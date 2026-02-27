"""
AniList API Scraper for Anime Review Context
=============================================
感情ベースのアニメ推薦アプリ用データ収集スクリプト。
AniList GraphQL APIを使用して人気アニメのレビュー・タグ情報を取得する。
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime

import requests

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# AniList API設定
ANILIST_API_URL = "https://graphql.anilist.co"
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "anime_reviews.json"

# レートリミット設定（AniListは90リクエスト/分）
RATE_LIMIT_REQUESTS = 85       # 安全マージンを取って85に設定
RATE_LIMIT_WINDOW_SEC = 60     # 1分間のウィンドウ
REQUEST_INTERVAL_SEC = RATE_LIMIT_WINDOW_SEC / RATE_LIMIT_REQUESTS  # ~0.7秒/リクエスト
RETRY_WAIT_SEC = 60            # レートリミット超過時の待機時間


# ─────────────────────────────────────────
# GraphQLクエリ定義
# ─────────────────────────────────────────

POPULAR_ANIME_QUERY = """
query PopularAnime($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title {
        romaji
        english
        native
      }
      description(asHtml: false)
      averageScore
      popularity
      genres
      tags {
        id
        name
        category
        rank
        isGeneralSpoiler
        isMediaSpoiler
      }
      startDate {
        year
      }
      status
      episodes
      format
    }
  }
}
"""

ANIME_REVIEWS_QUERY = """
query AnimeReviews($mediaId: Int, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
    }
    reviews(mediaId: $mediaId, sort: RATING_DESC) {
      id
      summary
      body(asHtml: false)
      rating
      ratingAmount
      score
      createdAt
    }
  }
}
"""


# ─────────────────────────────────────────
# APIクライアント
# ─────────────────────────────────────────

class RateLimiter:
    """シンプルなスライディングウィンドウ方式のレートリミッター。"""

    def __init__(self, max_requests: int, window_sec: float):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.request_timestamps: list[float] = []

    def wait_if_needed(self) -> None:
        now = time.time()
        # ウィンドウ外の古いタイムスタンプを削除
        self.request_timestamps = [
            ts for ts in self.request_timestamps if now - ts < self.window_sec
        ]
        if len(self.request_timestamps) >= self.max_requests:
            oldest = self.request_timestamps[0]
            sleep_sec = self.window_sec - (now - oldest) + 0.1
            logger.warning("レートリミット接近: %.1f秒待機します...", sleep_sec)
            time.sleep(max(sleep_sec, 0))

        self.request_timestamps.append(time.time())


class AniListClient:
    """AniList GraphQL APIクライアント。"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC)

    def _post(self, query: str, variables: dict, max_retries: int = 3) -> dict:
        """GraphQLリクエストを送信する（リトライ付き）。"""
        self.rate_limiter.wait_if_needed()

        payload = {"query": query, "variables": variables}

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    ANILIST_API_URL, json=payload, timeout=30
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", RETRY_WAIT_SEC))
                    logger.warning(
                        "レートリミット(429)に到達: %d秒後にリトライします...", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code == 500:
                    logger.warning("サーバーエラー(500): attempt %d/%d", attempt + 1, max_retries)
                    time.sleep(2 ** attempt)
                    continue

                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    logger.error("GraphQLエラー: %s", data["errors"])
                    return {}

                return data.get("data", {})

            except requests.exceptions.RequestException as exc:
                logger.error("リクエストエラー (attempt %d/%d): %s", attempt + 1, max_retries, exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("最大リトライ回数に達しました")
        return {}

    def fetch_popular_anime(self, total: int = 100, per_page: int = 50) -> list[dict]:
        """人気アニメの一覧を取得する。"""
        all_media = []
        page = 1

        while len(all_media) < total:
            remaining = total - len(all_media)
            fetch_count = min(per_page, remaining)

            logger.info("アニメ一覧取得中: page=%d (累計%d/%d件)", page, len(all_media), total)

            data = self._post(
                POPULAR_ANIME_QUERY,
                {"page": page, "perPage": fetch_count},
            )

            page_data = data.get("Page", {})
            media_list = page_data.get("media", [])
            page_info = page_data.get("pageInfo", {})

            if not media_list:
                logger.warning("アニメデータが空です。取得を終了します")
                break

            all_media.extend(media_list)
            logger.info("取得済み: %d件", len(all_media))

            if not page_info.get("hasNextPage", False):
                break

            page += 1

        return all_media[:total]

    def fetch_reviews_for_anime(
        self, media_id: int, max_reviews: int = 10
    ) -> list[dict]:
        """指定アニメのレビューを取得する（最大max_reviews件）。"""
        all_reviews = []
        page = 1
        per_page = min(max_reviews, 10)  # AniListの上限に合わせる

        while len(all_reviews) < max_reviews:
            data = self._post(
                ANIME_REVIEWS_QUERY,
                {"mediaId": media_id, "page": page, "perPage": per_page},
            )

            page_data = data.get("Page", {})
            reviews = page_data.get("reviews", [])
            page_info = page_data.get("pageInfo", {})

            if not reviews:
                break

            all_reviews.extend(reviews)

            if not page_info.get("hasNextPage", False) or len(all_reviews) >= max_reviews:
                break

            page += 1

        return all_reviews[:max_reviews]


# ─────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────

def build_anime_record(anime: dict, reviews: list[dict]) -> dict:
    """アニメ情報とレビューを1つのレコードにまとめる。"""
    tags = [
        {
            "id": tag["id"],
            "name": tag["name"],
            "category": tag["category"],
            "rank": tag["rank"],
            "is_general_spoiler": tag["isGeneralSpoiler"],
            "is_media_spoiler": tag["isMediaSpoiler"],
        }
        for tag in anime.get("tags", [])
        if not tag.get("isMediaSpoiler", False)  # 重大ネタバレタグは除外
    ]

    cleaned_reviews = [
        {
            "id": r["id"],
            "summary": r.get("summary") or "",
            "body": r.get("body") or "",
            "rating": r.get("rating"),
            "rating_amount": r.get("ratingAmount"),
            "score": r.get("score"),
            "created_at": r.get("createdAt"),
        }
        for r in reviews
        if r.get("body") or r.get("summary")  # 本文またはサマリーが存在するもののみ
    ]

    start_year = None
    if anime.get("startDate") and anime["startDate"].get("year"):
        start_year = anime["startDate"]["year"]

    return {
        "id": anime["id"],
        "title": {
            "romaji": anime["title"].get("romaji"),
            "english": anime["title"].get("english"),
            "native": anime["title"].get("native"),
        },
        "description": anime.get("description") or "",
        "average_score": anime.get("averageScore"),
        "popularity": anime.get("popularity"),
        "genres": anime.get("genres", []),
        "tags": tags,
        "start_year": start_year,
        "status": anime.get("status"),
        "episodes": anime.get("episodes"),
        "format": anime.get("format"),
        "reviews": cleaned_reviews,
        "review_count": len(cleaned_reviews),
    }


def scrape(total_anime: int = 100, max_reviews_per_anime: int = 10) -> None:
    """メインのスクレイピング処理。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AniListClient()
    results = []

    # ── Step 1: 人気アニメ一覧を取得 ──
    logger.info("=== Step 1: 人気アニメ一覧を取得 (上位%d件) ===", total_anime)
    anime_list = client.fetch_popular_anime(total=total_anime)
    logger.info("取得完了: %d件のアニメ", len(anime_list))

    # ── Step 2: 各アニメのレビューを取得 ──
    logger.info("=== Step 2: 各アニメのレビューを取得 ===")
    for idx, anime in enumerate(anime_list, start=1):
        media_id = anime["id"]
        title = anime["title"].get("romaji") or anime["title"].get("native") or str(media_id)

        logger.info("[%d/%d] レビュー取得中: %s (id=%d)", idx, len(anime_list), title, media_id)

        reviews = client.fetch_reviews_for_anime(
            media_id=media_id, max_reviews=max_reviews_per_anime
        )
        logger.info("  → %d件のレビューを取得", len(reviews))

        record = build_anime_record(anime, reviews)
        results.append(record)

        # 進捗の途中保存（10件ごと）
        if idx % 10 == 0:
            _save_json(results, OUTPUT_FILE)
            logger.info("途中保存: %s (%d件)", OUTPUT_FILE, len(results))

    # ── Step 3: 最終保存 ──
    _save_json(results, OUTPUT_FILE)
    _save_summary(results)

    logger.info("=== 完了: %d件のアニメデータを %s に保存しました ===", len(results), OUTPUT_FILE)


def _save_json(data: list[dict], path: Path) -> None:
    """JSONファイルに保存する。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "total_anime": len(data),
                    "source": "AniList GraphQL API",
                },
                "anime": data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _save_summary(data: list[dict]) -> None:
    """統計サマリーをログに出力する。"""
    total_reviews = sum(r["review_count"] for r in data)
    with_reviews = sum(1 for r in data if r["review_count"] > 0)
    logger.info(
        "サマリー: アニメ%d件 | レビューあり%d件 | 総レビュー数%d件",
        len(data),
        with_reviews,
        total_reviews,
    )


# ─────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AniList APIからアニメのレビュー・タグ情報を取得するスクリプト"
    )
    parser.add_argument(
        "--total-anime",
        type=int,
        default=100,
        help="取得するアニメ作品数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=10,
        help="1作品あたりの最大レビュー取得数 (デフォルト: 10)",
    )
    args = parser.parse_args()

    scrape(total_anime=args.total_anime, max_reviews_per_anime=args.max_reviews)

"""
Anime Sentiment Analyzer
========================
LLM（Claude）を使用してアニメレビューから感情スコアを抽出するスクリプト。
data/anime_reviews.json を読み込み、各作品の感情分析結果を
data/anime_with_sentiments.json として保存する。

感情カテゴリー（各 0.0〜1.0）:
  - tear_jerker : 涙活（悲しい・感動・切ない）
  - hype        : アドレナリン（熱い・興奮・燃える）
  - healing     : 癒やし（ほのぼの・日常・安心）
  - dark        : 衝撃/鬱（絶望・衝撃展開・シリアス）
  - comedy      : 爆笑（笑える・ギャグ・明るい）

使用方法:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python analyzer/sentiment_analyzer.py [--input data/anime_reviews.json]
                                         [--output data/anime_with_sentiments.json]
                                         [--concurrency 5]
"""

import asyncio
import json
import logging
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import anthropic

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────

DEFAULT_INPUT  = Path("data/anime_reviews.json")
DEFAULT_OUTPUT = Path("data/anime_with_sentiments.json")

# 同時処理数（Anthropic APIのレートリミット対策）
DEFAULT_CONCURRENCY = 5

# 使用するClaudeモデル
MODEL = "claude-haiku-4-5-20251001"  # 大量処理向けにHaikuを使用

# チェックポイント保存間隔（件数）
CHECKPOINT_INTERVAL = 10

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analyzer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# ツール定義（構造化出力のため tool_use を使用）
# ─────────────────────────────────────────

SENTIMENT_TOOL: dict = {
    "name": "record_sentiment",
    "description": (
        "Record the emotional sentiment analysis result for an anime, "
        "based on its reviews and description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tear_jerker": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0 for 涙活 (tear-jerker): "
                    "sad, emotional, moving, bittersweet moments."
                ),
            },
            "hype": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0 for アドレナリン (hype/excitement): "
                    "intense action, thrilling battles, pumped-up moments."
                ),
            },
            "healing": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0 for 癒やし (healing/cozy): "
                    "heartwarming, slice-of-life, relaxing, comforting."
                ),
            },
            "dark": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0 for 衝撃/鬱 (dark/shocking): "
                    "despair, plot twists, heavy themes, psychological depth."
                ),
            },
            "comedy": {
                "type": "number",
                "description": (
                    "Score 0.0-1.0 for 爆笑 (comedy): "
                    "funny, gag-heavy, lighthearted, laugh-out-loud moments."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Brief explanation (2-4 sentences) of why these scores were assigned, "
                    "citing specific themes or review excerpts."
                ),
            },
        },
        "required": ["tear_jerker", "hype", "healing", "dark", "comedy", "reasoning"],
    },
}


# ─────────────────────────────────────────
# プロンプト構築
# ─────────────────────────────────────────

def _build_prompt(anime: dict) -> str:
    """感情分析プロンプトを構築する。"""
    title_romaji = anime["title"].get("romaji") or ""
    title_english = anime["title"].get("english") or ""
    title_display = title_english or title_romaji or f"ID:{anime['id']}"

    genres = ", ".join(anime.get("genres", [])) or "N/A"
    tags = ", ".join(t["name"] for t in anime.get("tags", [])[:15]) or "N/A"
    description = (anime.get("description") or "").strip()[:500]  # 先頭500文字に絞る

    # レビュー本文を結合（長すぎる場合は末尾を切る）
    review_blocks: list[str] = []
    for i, review in enumerate(anime.get("reviews", []), start=1):
        summary = (review.get("summary") or "").strip()
        body = (review.get("body") or "").strip()[:800]  # レビュー本文は800文字まで
        score = review.get("score")
        parts = []
        if summary:
            parts.append(f"Summary: {summary}")
        if body:
            parts.append(f"Body: {body}")
        if score is not None:
            parts.append(f"Score: {score}/100")
        if parts:
            review_blocks.append(f"[Review {i}]\n" + "\n".join(parts))

    reviews_text = "\n\n".join(review_blocks) if review_blocks else "No reviews available."

    return f"""You are an expert anime sentiment analyst. Analyze the following anime and assign emotional scores.

## Anime: {title_display}
- Genres: {genres}
- Tags: {tags}
- Description: {description}

## User Reviews:
{reviews_text}

---
Analyze the emotional profile of this anime based on the reviews, genres, and description above.
Use the `record_sentiment` tool to submit your analysis.
All scores must be between 0.0 and 1.0. Scores can co-exist (e.g., an anime can be both healing AND have dark moments).
"""


# ─────────────────────────────────────────
# 感情分析コア
# ─────────────────────────────────────────

class SentimentAnalyzer:
    """Claude APIを使った非同期感情分析クライアント。"""

    def __init__(self, api_key: str, concurrency: int = DEFAULT_CONCURRENCY):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.semaphore = asyncio.Semaphore(concurrency)

    async def analyze_anime(self, anime: dict) -> dict:
        """1作品の感情スコアを取得する。"""
        title = (
            anime["title"].get("english")
            or anime["title"].get("romaji")
            or str(anime["id"])
        )

        # レビューも説明文もない場合はスキップ
        has_content = anime.get("reviews") or anime.get("description")
        if not has_content:
            logger.warning("スキップ (コンテンツなし): %s", title)
            return _make_skipped_record(anime, reason="no_content")

        async with self.semaphore:
            return await self._call_api(anime, title)

    async def _call_api(self, anime: dict, title: str, max_retries: int = 3) -> dict:
        """Claude APIを呼び出してセンチメントを取得する（リトライ付き）。"""
        prompt = _build_prompt(anime)
        wait = 2

        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    tools=[SENTIMENT_TOOL],
                    tool_choice={"type": "tool", "name": "record_sentiment"},
                    messages=[{"role": "user", "content": prompt}],
                )

                # tool_use ブロックからスコアを取り出す
                for block in response.content:
                    if block.type == "tool_use" and block.name == "record_sentiment":
                        scores = block.input
                        return _make_result_record(anime, scores)

                logger.error("tool_use ブロックが見つかりません: %s", title)
                return _make_skipped_record(anime, reason="no_tool_use_block")

            except anthropic.RateLimitError:
                logger.warning("レートリミット: %d秒待機後リトライ (attempt %d)", wait, attempt + 1)
                await asyncio.sleep(wait)
                wait *= 2

            except anthropic.APIStatusError as exc:
                logger.error("APIエラー %s (attempt %d): %s", exc.status_code, attempt + 1, exc.message)
                if exc.status_code >= 500:
                    await asyncio.sleep(wait)
                    wait *= 2
                else:
                    return _make_skipped_record(anime, reason=f"api_error_{exc.status_code}")

            except Exception as exc:  # noqa: BLE001
                logger.error("予期しないエラー (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(wait)
                wait *= 2

        return _make_skipped_record(anime, reason="max_retries_exceeded")


# ─────────────────────────────────────────
# レコード構築ヘルパー
# ─────────────────────────────────────────

def _clamp(value: float) -> float:
    """スコアを 0.0〜1.0 にクランプする。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _make_result_record(anime: dict, scores: dict) -> dict:
    """分析成功レコードを生成する。"""
    sentiment = {
        "tear_jerker": _clamp(scores.get("tear_jerker", 0.0)),
        "hype":        _clamp(scores.get("hype", 0.0)),
        "healing":     _clamp(scores.get("healing", 0.0)),
        "dark":        _clamp(scores.get("dark", 0.0)),
        "comedy":      _clamp(scores.get("comedy", 0.0)),
        "reasoning":   str(scores.get("reasoning", "")),
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "status":      "ok",
    }
    return {**anime, "sentiment": sentiment}


def _make_skipped_record(anime: dict, reason: str) -> dict:
    """スキップ/エラーレコードを生成する。"""
    sentiment = {
        "tear_jerker": None,
        "hype":        None,
        "healing":     None,
        "dark":        None,
        "comedy":      None,
        "reasoning":   None,
        "analyzed_at": datetime.utcnow().isoformat() + "Z",
        "status":      f"skipped:{reason}",
    }
    return {**anime, "sentiment": sentiment}


# ─────────────────────────────────────────
# I/O ヘルパー
# ─────────────────────────────────────────

def load_input(path: Path) -> tuple[dict, list[dict]]:
    """入力JSONを読み込む。(metadata, anime_list) を返す。"""
    if not path.exists():
        logger.error("入力ファイルが見つかりません: %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("metadata", {}), raw.get("anime", [])


def save_output(results: list[dict], source_metadata: dict, output_path: Path) -> None:
    """分析結果をJSONに保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok_count = sum(1 for r in results if r.get("sentiment", {}).get("status") == "ok")
    payload = {
        "metadata": {
            **source_metadata,
            "sentiment_model": MODEL,
            "sentiment_generated_at": datetime.utcnow().isoformat() + "Z",
            "total_analyzed": ok_count,
            "total_skipped": len(results) - ok_count,
        },
        "anime": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("保存完了: %s (%d件)", output_path, len(results))


# ─────────────────────────────────────────
# メイン非同期処理
# ─────────────────────────────────────────

async def run(
    input_path: Path,
    output_path: Path,
    concurrency: int,
    api_key: str,
) -> None:
    source_metadata, anime_list = load_input(input_path)
    total = len(anime_list)
    logger.info("=== 感情分析開始: %d件のアニメ (並行数=%d) ===", total, concurrency)

    analyzer = SentimentAnalyzer(api_key=api_key, concurrency=concurrency)
    results: list[dict] = []
    completed = 0

    # タスクを生成し、完了した順に処理する（asyncio.as_completed）
    tasks = {
        asyncio.create_task(analyzer.analyze_anime(anime), name=str(anime["id"])): anime
        for anime in anime_list
    }

    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        completed += 1

        title = (
            result["title"].get("english")
            or result["title"].get("romaji")
            or str(result["id"])
        )
        status = result.get("sentiment", {}).get("status", "?")
        logger.info(
            "[%d/%d] %s — %s",
            completed,
            total,
            title,
            status,
        )

        # チェックポイント保存
        if completed % CHECKPOINT_INTERVAL == 0:
            save_output(results, source_metadata, output_path)
            logger.info("チェックポイント保存: %d件処理済み", completed)

    # 最終保存
    save_output(results, source_metadata, output_path)
    logger.info("=== 完了: ok=%d / skip=%d ===",
                sum(1 for r in results if r.get("sentiment", {}).get("status") == "ok"),
                sum(1 for r in results if r.get("sentiment", {}).get("status", "").startswith("skipped")))


# ─────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claudeを使ってアニメレビューの感情分析を行うスクリプト"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"入力JSONファイルパス (デフォルト: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"出力JSONファイルパス (デフォルト: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"同時APIリクエスト数 (デフォルト: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error(
            "環境変数 ANTHROPIC_API_KEY が設定されていません。\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...' を実行してください。"
        )
        sys.exit(1)

    asyncio.run(
        run(
            input_path=args.input,
            output_path=args.output,
            concurrency=args.concurrency,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    main()

"""
Anime Chat UI
=============
Claudeとの対話を通じてユーザーの気分を深掘りし、
感情ベクトルを生成して最適なアニメを推薦するチャットインターフェース。

処理フロー:
  Phase 1 │ 深掘り対話 (3ターン)
          ↓
  Phase 2 │ 対話内容 → 感情ベクトル抽出 (tool_use)
          ↓
  Phase 3 │ AnimeRecommender で上位5件を取得
          ↓
  Phase 4 │ 対話文脈に基づくマッチ理由を生成
          ↓
  Phase 5 │ リッチ表示

使用方法:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python interface/chat_ui.py [--data data/anime_with_sentiments.json] [--top-k 5]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import anthropic

# プロジェクトルートを import パスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.recommender import (
    EMOTION_KEYS,
    EMOTION_LABELS,
    AnimeRecommender,
    Recommendation,
)

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────

DEFAULT_DATA_PATH = Path("data/anime_with_sentiments.json")
MODEL             = "claude-sonnet-4-6"   # 対話品質重視でSonnetを使用
TOP_K             = 5
DIALOGUE_TURNS    = 3                     # 深掘りターン数

# ─────────────────────────────────────────
# システムプロンプト
# ─────────────────────────────────────────

_CONVERSATION_SYSTEM = """\
あなたは親しみやすいアニメ推薦アシスタントです。
ユーザーと自然な会話を通じて、今どんなアニメを見たいのかを理解します。

【会話のルール】
- 1回のメッセージは3〜5文程度の短さを保つ。
- 質問は1ターンに1つだけ。複数の質問を一度にしない。
- ユーザーの言葉を受け止め、共感を示してから次の質問へ進む。
- アニメ用語や固有名詞を使った自然な返答をする。
- 感情スコアを直接聞かない。あくまで自然な会話で気持ちを引き出す。

【ターン設計】
- Turn 1: 「最近見て印象に残ったアニメ」または「今どんな気分でアニメを見たいか」を聞く。
- Turn 2: Turn 1の回答を受けて、感情・雰囲気の方向性を深掘りする。
  （例: 泣きたい気分なら「どんな場面で泣きたい？」、アクションなら「どんな熱さが好き？」）
- Turn 3: 好みをさらに絞り込む最後の質問をする。
  （例: 「明るい終わり方が好き？それとも余韻が残る感じ？」）
このターン設計に従ってください。会話は必ず3ターンで終わります。"""

_EXTRACTION_SYSTEM = """\
あなたはアニメの感情分析の専門家です。
提供されたユーザーとの会話ログを分析し、
ユーザーが求めているアニメの感情プロファイルを数値化します。"""

_MATCH_REASON_SYSTEM = """\
あなたはアニメ推薦の説明担当AIです。
ユーザーとの会話内容と、推薦されたアニメの感情プロファイルをもとに、
「なぜこの会話からこのアニメを選んだのか」を日本語で簡潔に説明してください。"""

# ─────────────────────────────────────────
# Tool 定義
# ─────────────────────────────────────────

_VECTOR_EXTRACTION_TOOL: dict = {
    "name": "extract_emotion_vector",
    "description": (
        "会話ログを分析し、ユーザーが求めるアニメの5次元感情ベクトルを抽出する。"
        "各スコアは 0.0〜1.0 の範囲で設定する。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tear_jerker": {
                "type": "number",
                "description": "涙活スコア: 感動・悲しみ・切なさへの欲求 (0.0〜1.0)",
            },
            "hype": {
                "type": "number",
                "description": "アドレナリンスコア: 熱さ・興奮・バトルへの欲求 (0.0〜1.0)",
            },
            "healing": {
                "type": "number",
                "description": "癒やしスコア: ほのぼの・日常・安心感への欲求 (0.0〜1.0)",
            },
            "dark": {
                "type": "number",
                "description": "衝撃/鬱スコア: シリアス・絶望・深いテーマへの欲求 (0.0〜1.0)",
            },
            "comedy": {
                "type": "number",
                "description": "爆笑スコア: 笑い・ギャグ・明るさへの欲求 (0.0〜1.0)",
            },
            "user_intent_summary": {
                "type": "string",
                "description": "ユーザーが求めているアニメ体験を日本語2〜3文で要約する。",
            },
        },
        "required": [
            "tear_jerker", "hype", "healing", "dark", "comedy",
            "user_intent_summary",
        ],
    },
}

_MATCH_REASONS_TOOL: dict = {
    "name": "provide_match_reasons",
    "description": "各推薦アニメについて、ユーザーとの会話に基づくマッチ理由を提供する。",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "各アニメのマッチ理由（日本語2〜3文）。"
                    "リストの順番は入力の推薦順と一致させること。"
                ),
            },
        },
        "required": ["reasons"],
    },
}

# ─────────────────────────────────────────
# 表示ユーティリティ
# ─────────────────────────────────────────

_W = 68  # 出力幅


def _hr(char: str = "─") -> None:
    print(char * _W)


def _box(text: str, char: str = "═") -> None:
    print(char * _W)
    print(f"  {text}")
    print(char * _W)


def _bar(value: float, width: int = 18) -> str:
    filled = round(value * width)
    return "▓" * filled + "░" * (width - filled)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _print_emotion_vector(vec: dict[str, float], summary: str) -> None:
    print()
    _box("感情ベクトル（あなたの今の気分）", "─")
    for key, label in EMOTION_LABELS.items():
        val = vec[key]
        print(f"  {label:<28}  {_bar(val)}  {val:.2f}")
    print()
    print(f"  要約: {summary}")
    print()


def _print_recommendations(
    results: list[Recommendation],
    match_reasons: list[str],
) -> None:
    print()
    _box(f"あなたへのアニメ推薦 TOP {len(results)}", "═")
    print()
    for rec, reason in zip(results, match_reasons):
        _hr()
        print(f"  #{rec.rank}  {rec.title}")
        if rec.genres:
            print(f"      ジャンル : {', '.join(rec.genres[:5])}")
        avg = f"{rec.average_score}/100" if rec.average_score else "N/A"
        pop = f"{rec.popularity_raw:,}" if rec.popularity_raw else "N/A"
        print(f"      評価     : {avg}   人気度: {pop}")

        print()
        print(f"      Final Score  {rec.final_score:.4f}  {_bar(rec.final_score)}")
        print(f"      感情一致度   {rec.cosine_sim:.4f}  {_bar(rec.cosine_sim)}")
        print(f"      人気ボーナス {rec.popularity_score:.4f}  {_bar(rec.popularity_score)}")

        print()
        print("      【この会話からこの作品を選んだ理由】")
        # 理由を60文字で折り返す
        words = reason.split()
        buf: list[str] = []
        line_len = 0
        for word in words:
            if line_len + len(word) + 1 > 58:
                print(f"      {' '.join(buf)}")
                buf, line_len = [word], len(word)
            else:
                buf.append(word)
                line_len += len(word) + 1
        if buf:
            print(f"      {' '.join(buf)}")
        print()

    _hr("═")
    print()


# ─────────────────────────────────────────
# 非同期入力
# ─────────────────────────────────────────

async def _ainput(prompt: str = "") -> str:
    """ブロッキング input() を ThreadPoolExecutor でラップした非同期版。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


# ─────────────────────────────────────────
# Phase 1: 深掘り対話
# ─────────────────────────────────────────

async def conduct_conversation(client: anthropic.AsyncAnthropic) -> list[dict]:
    """
    3ターンの深掘り対話を行い、会話ログ（messages）を返す。
    messages は Claude API の role/content 形式。
    """
    print()
    _box("アニメ推薦チャット  ─  会話モード", "═")
    print("  ※ 3回の質問でぴったりな作品を探します。")
    print("  ※ 終了するには Ctrl+C を押してください。")
    print()

    messages: list[dict] = []

    for turn in range(1, DIALOGUE_TURNS + 1):
        # アシスタントの返答を取得
        response = await client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=_CONVERSATION_SYSTEM,
            messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        # ターン表示
        _hr()
        print(f"  AI  [{turn}/{DIALOGUE_TURNS}]\n")
        print(f"  {assistant_text}")
        print()

        # ユーザー入力
        user_text = (await _ainput("  あなた > ")).strip()
        while not user_text:
            user_text = (await _ainput("  あなた > ")).strip()

        messages.append({"role": "user", "content": user_text})
        print()

    return messages


# ─────────────────────────────────────────
# Phase 2: 感情ベクトル抽出
# ─────────────────────────────────────────

def _format_conversation_for_extraction(messages: list[dict]) -> str:
    """会話ログを抽出プロンプト用のテキストに変換する。"""
    lines = ["以下はユーザーとの会話ログです。\n"]
    for msg in messages:
        role = "アシスタント" if msg["role"] == "assistant" else "ユーザー"
        lines.append(f"[{role}]: {msg['content']}")
    return "\n".join(lines)


async def extract_emotion_vector(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
) -> tuple[dict[str, float], str]:
    """
    会話ログから感情ベクトルと意図サマリーを抽出する。
    Returns: (vector_dict, user_intent_summary)
    """
    print("  感情ベクトルを分析中...")

    conversation_text = _format_conversation_for_extraction(messages)
    prompt = (
        f"{conversation_text}\n\n"
        "上記の会話をもとに、ユーザーが今求めているアニメの感情プロファイルを抽出してください。"
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_EXTRACTION_SYSTEM,
        tools=[_VECTOR_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_emotion_vector"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_emotion_vector":
            raw = block.input
            vector = {k: _clamp(raw.get(k, 0.0)) for k in EMOTION_KEYS}
            summary = str(raw.get("user_intent_summary", ""))
            return vector, summary

    # フォールバック（ツール呼び出し失敗時）
    return {k: 0.2 for k in EMOTION_KEYS}, "抽出に失敗しました。"


# ─────────────────────────────────────────
# Phase 4: マッチ理由の生成
# ─────────────────────────────────────────

def _build_match_reason_prompt(
    messages: list[dict],
    user_intent_summary: str,
    results: list[Recommendation],
) -> str:
    """マッチ理由生成用プロンプトを構築する。"""
    conv = _format_conversation_for_extraction(messages)

    anime_list_text = "\n".join(
        f"{i+1}. {r.title} (ジャンル: {', '.join(r.genres[:3])}) — 感情分析サマリー: {r.reasoning[:150]}"
        for i, r in enumerate(results)
    )

    return (
        f"{conv}\n\n"
        f"【ユーザーの求める体験の要約】\n{user_intent_summary}\n\n"
        f"【推薦されたアニメ一覧】\n{anime_list_text}\n\n"
        "各アニメについて、上記の会話内容をもとに「なぜこの対話からこの作品を選んだのか」"
        "を日本語2〜3文で説明してください。"
        f"リストは{len(results)}件です。順番を変えずに provide_match_reasons ツールで返してください。"
    )


async def generate_match_reasons(
    client: anthropic.AsyncAnthropic,
    messages: list[dict],
    user_intent_summary: str,
    results: list[Recommendation],
) -> list[str]:
    """
    推薦結果ごとに、対話文脈に基づくマッチ理由を生成する。
    """
    print("  マッチ理由を生成中...")

    prompt = _build_match_reason_prompt(messages, user_intent_summary, results)

    response = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_MATCH_REASON_SYSTEM,
        tools=[_MATCH_REASONS_TOOL],
        tool_choice={"type": "tool", "name": "provide_match_reasons"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "provide_match_reasons":
            reasons: list[str] = block.input.get("reasons", [])
            # 件数が足りない場合はフォールバック
            while len(reasons) < len(results):
                reasons.append("この作品はあなたの感情プロファイルと高い類似度を持っています。")
            return reasons[: len(results)]

    return ["マッチ理由の生成に失敗しました。"] * len(results)


# ─────────────────────────────────────────
# メインオーケストレーター
# ─────────────────────────────────────────

async def run(
    data_path: Path,
    top_k: int,
    api_key: str,
) -> None:
    # 推薦エンジンをロード（同期処理）
    try:
        recommender = AnimeRecommender(data_path=data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nエラー: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  推薦データベース: {recommender.loaded_count}件の作品をロード済み")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Phase 1: 深掘り対話 ──────────────────
    try:
        conversation = await conduct_conversation(client)
    except (KeyboardInterrupt, EOFError):
        print("\n\n  会話を中断しました。")
        return

    # ── Phase 2: 感情ベクトル抽出 ────────────
    _hr()
    print()
    vector, user_intent_summary = await extract_emotion_vector(client, conversation)
    _print_emotion_vector(vector, user_intent_summary)

    # ── Phase 3: 推薦エンジン呼び出し ────────
    print("  推薦作品を検索中...")
    try:
        results = recommender.get_recommendations(user_vector=vector, top_k=top_k)
    except ValueError as exc:
        print(f"\nエラー: {exc}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("\n推薦できる作品が見つかりませんでした。")
        return

    # ── Phase 4: マッチ理由の生成 ─────────────
    match_reasons = await generate_match_reasons(
        client, conversation, user_intent_summary, results
    )

    # ── Phase 5: リッチ表示 ───────────────────
    _print_recommendations(results, match_reasons)


# ─────────────────────────────────────────
# エントリーポイント
# ─────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Claudeと対話してアニメを推薦するチャットUI"
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
        default=TOP_K,
        help=f"推薦件数 (デフォルト: {TOP_K})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "エラー: 環境変数 ANTHROPIC_API_KEY が設定されていません。\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...' を実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(run(data_path=args.data, top_k=args.top_k, api_key=api_key))


if __name__ == "__main__":
    main()

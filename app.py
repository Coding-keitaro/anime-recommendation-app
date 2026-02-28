"""
アニメ感情推薦チャットアプリ
============================
Streamlit + Claude を使ったアニメ推薦チャットUI。
3ターンの深掘り対話で感情ベクトルを生成し、上位5件を推薦する。

起動方法:
  export ANTHROPIC_API_KEY="sk-ant-..."
  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

# engine/ を import パスに追加
sys.path.insert(0, str(Path(__file__).parent))

from engine.recommender import (
    EMOTION_KEYS,
    EMOTION_LABELS,
    AnimeRecommender,
    Recommendation,
)

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────

MODEL = 'claude-3-5-sonnet-latest'
TOP_K          = 5
DIALOGUE_TURNS = 3
DATA_PATH      = Path("data/anime_with_sentiments.json")
ANILIST_URL    = "https://anilist.co/anime"

# 感情ラベルの絵文字マッピング
EMOTION_EMOJI: dict[str, str] = {
    "tear_jerker": "😢",
    "hype":        "🔥",
    "healing":     "🌸",
    "dark":        "🌑",
    "comedy":      "😂",
}

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

【ターン設計（厳守）】
- Turn 1: 「最近見て印象に残ったアニメ」または「今どんな気分でアニメを見たいか」を聞く。
- Turn 2: Turn 1の回答を受けて、感情・雰囲気の方向性を深掘りする。
- Turn 3: 好みをさらに絞り込む最後の質問をする。終盤には「ではおすすめを探してみますね！」という期待感を持たせる一言を添える。"""

_EXTRACTION_SYSTEM = """\
あなたはアニメの感情分析の専門家です。
提供されたユーザーとの会話ログを分析し、
ユーザーが求めているアニメの感情プロファイルを数値化します。"""

_MATCH_REASON_SYSTEM = """\
あなたはアニメ推薦の説明担当AIです。
ユーザーとの会話内容と推薦アニメの感情プロファイルをもとに、
「なぜこの会話からこのアニメを選んだのか」を日本語で簡潔に説明してください。"""

# ─────────────────────────────────────────
# Tool 定義
# ─────────────────────────────────────────

_VECTOR_TOOL: dict = {
    "name": "extract_emotion_vector",
    "description": "会話ログを分析してユーザーが求める5次元感情ベクトルを抽出する。",
    "input_schema": {
        "type": "object",
        "properties": {
            "tear_jerker": {"type": "number", "description": "涙活スコア 0.0〜1.0"},
            "hype":        {"type": "number", "description": "アドレナリンスコア 0.0〜1.0"},
            "healing":     {"type": "number", "description": "癒やしスコア 0.0〜1.0"},
            "dark":        {"type": "number", "description": "衝撃/鬱スコア 0.0〜1.0"},
            "comedy":      {"type": "number", "description": "爆笑スコア 0.0〜1.0"},
            "user_intent_summary": {
                "type": "string",
                "description": "ユーザーが求めるアニメ体験を日本語2〜3文で要約する。",
            },
        },
        "required": ["tear_jerker", "hype", "healing", "dark", "comedy", "user_intent_summary"],
    },
}

_MATCH_REASONS_TOOL: dict = {
    "name": "provide_match_reasons",
    "description": "各推薦アニメについて会話に基づくマッチ理由を提供する。",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "各アニメのマッチ理由（日本語2〜3文）。推薦順と一致させること。",
            },
        },
        "required": ["reasons"],
    },
}

# ─────────────────────────────────────────
# キャッシュされたリソース
# ─────────────────────────────────────────

@st.cache_resource
def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error(
            "⚠️ **ANTHROPIC_API_KEY** が設定されていません。\n\n"
            "```\nexport ANTHROPIC_API_KEY='sk-ant-...'\nstreamlit run app.py\n```"
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def _get_recommender() -> AnimeRecommender | None:
    try:
        return AnimeRecommender(data_path=DATA_PATH)
    except (FileNotFoundError, ValueError):
        return None


# ─────────────────────────────────────────
# セッション状態の初期化
# ─────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "messages":             [],     # [{role, content}]
        "emotion_vector":       None,   # dict | None
        "user_intent_summary":  "",
        "recommendations":      None,   # list[Recommendation] | None
        "match_reasons":        None,   # list[str] | None
        "phase":                "chatting",  # "chatting" | "done"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────
# LLM 呼び出し（同期）
# ─────────────────────────────────────────

def _stream_ai_response(messages: list[dict]) -> str:
    """会話ターン用。ストリーミングで AI の返答を表示し、全文を返す。"""
    client = _get_client()
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=300,
        system=_CONVERSATION_SYSTEM,
        messages=messages,
    ) as stream:
        full_text = st.write_stream(stream.text_stream)
    return full_text


def _extract_emotion_vector(messages: list[dict]) -> tuple[dict[str, float], str]:
    """会話ログから感情ベクトルと意図サマリーを抽出する。"""
    client = _get_client()

    lines = ["以下はユーザーとの会話ログです。\n"]
    for msg in messages:
        role = "アシスタント" if msg["role"] == "assistant" else "ユーザー"
        lines.append(f"[{role}]: {msg['content']}")
    conv_text = "\n".join(lines)

    prompt = f"{conv_text}\n\n上記の会話をもとに感情プロファイルを抽出してください。"
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_EXTRACTION_SYSTEM,
        tools=[_VECTOR_TOOL],
        tool_choice={"type": "tool", "name": "extract_emotion_vector"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_emotion_vector":
            raw = block.input
            vector = {k: max(0.0, min(1.0, float(raw.get(k, 0.0)))) for k in EMOTION_KEYS}
            summary = str(raw.get("user_intent_summary", ""))
            return vector, summary

    return {k: 0.2 for k in EMOTION_KEYS}, "感情分析中..."


def _generate_match_reasons(
    messages: list[dict],
    user_intent_summary: str,
    results: list[Recommendation],
) -> list[str]:
    """推薦結果ごとに対話文脈に基づくマッチ理由を生成する。"""
    client = _get_client()

    lines = ["以下はユーザーとの会話ログです。\n"]
    for msg in messages:
        role = "アシスタント" if msg["role"] == "assistant" else "ユーザー"
        lines.append(f"[{role}]: {msg['content']}")
    conv_text = "\n".join(lines)

    anime_list_text = "\n".join(
        f"{i+1}. {r.title} (ジャンル: {', '.join(r.genres[:3])}) — {r.reasoning[:120]}"
        for i, r in enumerate(results)
    )

    prompt = (
        f"{conv_text}\n\n"
        f"【ユーザーの求める体験】\n{user_intent_summary}\n\n"
        f"【推薦アニメ一覧】\n{anime_list_text}\n\n"
        "各アニメについて「なぜこの会話からこの作品を選んだのか」を"
        f"日本語2〜3文で説明してください。{len(results)}件、順番通りに返してください。"
    )

    response = client.messages.create(
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
            while len(reasons) < len(results):
                reasons.append("この作品はあなたの感情プロファイルと高い類似度を持っています。")
            return reasons[: len(results)]

    return ["マッチ理由の生成に失敗しました。"] * len(results)


# ─────────────────────────────────────────
# サイドバー
# ─────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.header("🎭 あなたの感情プロファイル")

        vec = st.session_state.emotion_vector
        user_turn = sum(1 for m in st.session_state.messages if m["role"] == "user")

        # ── 感情バーチャート ──────────────────
        if vec:
            labels = [
                f"{EMOTION_EMOJI[k]} {EMOTION_LABELS[k].split(' ')[0]}"
                for k in EMOTION_KEYS
            ]
            df = pd.DataFrame(
                {"スコア": [round(vec[k], 2) for k in EMOTION_KEYS]},
                index=labels,
            )
            st.bar_chart(df, height=220, use_container_width=True)

            # 最高スコアの感情をハイライト
            top_key = max(vec, key=lambda k: vec[k])
            top_label = EMOTION_LABELS[top_key]
            st.caption(f"🏆 最も強い感情: **{top_label}**")
        else:
            # プレースホルダー
            labels = [
                f"{EMOTION_EMOJI[k]} {EMOTION_LABELS[k].split(' ')[0]}"
                for k in EMOTION_KEYS
            ]
            df = pd.DataFrame({"スコア": [0.0] * 5}, index=labels)
            st.bar_chart(df, height=220, use_container_width=True)
            st.caption("✨ 対話を進めると感情スコアが更新されます")

        # ── 意図サマリー ──────────────────────
        if st.session_state.user_intent_summary:
            st.info(f"💭 {st.session_state.user_intent_summary}")

        st.divider()

        # ── 対話進捗 ──────────────────────────
        st.caption("対話進捗")
        st.progress(
            user_turn / DIALOGUE_TURNS,
            text=f"Turn {user_turn} / {DIALOGUE_TURNS}",
        )

        st.divider()

        # ── リセットボタン ────────────────────
        if st.button("🔄 最初からやり直す", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────
# 推薦結果の表示
# ─────────────────────────────────────────

def _render_recommendations() -> None:
    results: list[Recommendation] = st.session_state.recommendations
    reasons: list[str] = st.session_state.match_reasons

    st.divider()
    st.subheader("🎬 あなたへのおすすめアニメ TOP 5")

    for rec, reason in zip(results, reasons):
        with st.container(border=True):
            col_info, col_score = st.columns([3, 2], gap="large")

            with col_info:
                # タイトルとランク
                st.markdown(f"### #{rec.rank}  {rec.title}")

                # ジャンルバッジ
                if rec.genres:
                    genre_md = "  ".join(
                        f"`{g}`" for g in rec.genres[:5]
                    )
                    st.markdown(genre_md)

                # マッチ理由
                st.markdown("**💡 この会話から選んだ理由**")
                st.write(reason)

                # AniList リンク
                anilist_url = f"{ANILIST_URL}/{rec.anime_id}"
                st.link_button("AniList で詳細を見る 🔗", anilist_url)

            with col_score:
                # メインスコアカード
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "🏅 総合スコア",
                        f"{rec.final_score:.3f}",
                        help="感情一致度 × 0.8 + 人気度 × 0.2",
                    )
                with col_b:
                    avg_display = f"{int(rec.average_score)}" if rec.average_score else "N/A"
                    st.metric("⭐ 評価", avg_display, help="AniList ユーザー評価 (/100)")

                # スコアバー
                st.progress(
                    rec.cosine_sim,
                    text=f"感情一致度: {rec.cosine_sim:.3f}",
                )
                st.progress(
                    rec.popularity_score,
                    text=f"人気スコア: {rec.popularity_score:.3f}",
                )

                # 人気度（raw）
                if rec.popularity_raw:
                    st.caption(f"👥 {rec.popularity_raw:,} ユーザーが登録")


# ─────────────────────────────────────────
# 分析フェーズ
# ─────────────────────────────────────────

def _run_analysis() -> None:
    """全ターン完了後に感情抽出→推薦→理由生成を実行する。"""
    msgs = st.session_state.messages

    with st.status("✨ あなたにぴったりのアニメを探しています...", expanded=True) as status:

        st.write("🧠 感情ベクトルを抽出中...")
        vec, summary = _extract_emotion_vector(msgs)
        st.session_state.emotion_vector       = vec
        st.session_state.user_intent_summary  = summary

        recommender = _get_recommender()
        if recommender is None:
            status.update(label="エラー: データファイルが見つかりません", state="error")
            st.error(
                f"**{DATA_PATH}** が見つかりません。\n\n"
                "`sentiment_analyzer.py` を先に実行してください。"
            )
            st.stop()

        st.write("🔍 おすすめ作品を検索中...")
        results = recommender.get_recommendations(user_vector=vec, top_k=TOP_K)
        st.session_state.recommendations = results

        st.write("💬 推薦理由を生成中...")
        reasons = _generate_match_reasons(msgs, summary, results)
        st.session_state.match_reasons = reasons

        st.session_state.phase = "done"
        status.update(label="✅ 完了！あなたへのおすすめが決まりました", state="complete")

    st.rerun()


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="アニメ感情推薦",
        page_icon="🎌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()
    _render_sidebar()

    # ── ヘッダー ────────────────────────────
    st.title("🎌 アニメ感情推薦チャット")
    st.caption(
        "今の気分を話してください。3つの質問であなたにぴったりのアニメを見つけます。"
    )

    msgs: list[dict] = st.session_state.messages
    user_turn_count = sum(1 for m in msgs if m["role"] == "user")

    # ── 既存メッセージを描画 ──────────────────
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── フェーズ分岐 ──────────────────────────

    # 推薦完了済み
    if st.session_state.phase == "done":
        _render_recommendations()
        return

    # 全ターン終了 → 分析実行
    if user_turn_count == DIALOGUE_TURNS:
        _run_analysis()
        return

    # AI のターン（偶数メッセージ数 = AI が次に話す）
    if len(msgs) > 0 and len(msgs) % 2 == 0:
        turn_label = f"Turn {user_turn_count + 1} / {DIALOGUE_TURNS}"
        with st.chat_message("assistant"):
            with st.spinner(f"考え中... ({turn_label})"):
                # ストリーミング表示
                pass
            ai_text = _stream_ai_response(msgs)

        msgs.append({"role": "assistant", "content": ai_text})

        # ターン2以降は部分的な感情ベクトルを更新してサイドバーに反映
        if user_turn_count >= 1:
            with st.spinner("感情プロファイルを更新中..."):
                vec, summary = _extract_emotion_vector(msgs)
            st.session_state.emotion_vector      = vec
            st.session_state.user_intent_summary = summary

        st.rerun()

    # ユーザーのターン（奇数メッセージ数 = ユーザーが入力）
    else:
        placeholders = [
            "アニメのことを教えてください...",
            "もう少し詳しく教えてください...",
            "最後の質問です...",
        ]
        placeholder = placeholders[min(user_turn_count, 2)]

        user_input: str | None = st.chat_input(placeholder)
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            msgs.append({"role": "user", "content": user_input})
            st.rerun()


if __name__ == "__main__":
    main()

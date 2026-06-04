"""
CyberTA – Gradio App
Single-page chat UI with Ollama model selector.
VDB must be pre-built with build_vdb.py.
"""

import os
import gradio as gr
from llm_providers import model_list, DEFAULT_MODEL

EMBED_MODELS = {
    "MiniLM-L6-v2 (fast, 384d)"         : "all-MiniLM-L6-v2",
    "MPNet-base-v2 (best quality, 768d)" : "all-mpnet-base-v2",
    "BGE-base-en-v1.5 (quality, 768d)"  : "BAAI/bge-base-en-v1.5",
}
DEFAULT_EMBED = "MiniLM-L6-v2 (fast, 384d)"

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import rag_pipeline as rp
        _pipeline = rp
    return _pipeline


def _stats_md():
    try:
        s = get_pipeline().collection_stats()
        if s["total_chunks"] == 0:
            return "⚠️ VDB empty — run `python build_vdb.py` first."
        return (f"✅ **{s['total_chunks']} chunks** from "
                f"**{len(s['sources'])} document(s)**: "
                + ", ".join(f"`{s}`" for s in sorted(s["sources"])))
    except Exception as e:
        return f"⚠️ {e}"


def handle_chat(user_msg, history, llm_name, embed_name):
    if not user_msg.strip():
        return history, ""

    try:
        rp = get_pipeline()
        s  = rp.collection_stats()
        if s["total_chunks"] == 0:
            reply = "⚠️ Knowledge base is empty. Run `python build_vdb.py` on Hamming first."
            history.append({"role": "user",      "content": user_msg})
            history.append({"role": "assistant",  "content": reply})
            return history, ""

        # Build clean plain-text history for RAG
        plain_history = []
        for turn in history:
            if not isinstance(turn, dict):
                continue
            role    = turn.get("role", "")
            content = turn.get("content", "")
            # Gradio 6 stores content as list of dicts
            if isinstance(content, list):
                content = " ".join(x.get("text", "") for x in content if isinstance(x, dict))
            # Strip injected RAG context from previous turns to keep history short
            if role == "user" and "Context:" in content and "Question:" in content:
                content = content.split("Question:")[-1].strip()
            if role in ("user", "assistant") and content.strip():
                plain_history.append({"role": role, "content": content})

        reply, sources = rp.rag_query(
            user_msg, plain_history,
            llm_display_name   = llm_name,
            embed_display_name = embed_name,
        )

        if sources:
            reply += f"\n\n> 📎 *Sources: {', '.join(sources)}*"

    except Exception as e:
        import traceback
        traceback.print_exc()
        reply = f"❌ {e}"

    history.append({"role": "user",      "content": user_msg})
    history.append({"role": "assistant",  "content": reply})
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────

TITLE = "🤖 CyberTA – AI Teaching Assistant"

CSS = """
footer { display: none !important; }
#header { text-align: center; padding: 10px 0 0 0; }
#stats  { font-size: 0.85rem; color: #888; padding: 4px 8px; }
#model-row { background: #1a1a2e; border-radius: 8px; padding: 8px 16px; }
"""

with gr.Blocks(title=TITLE, css=CSS) as demo:

    gr.Markdown(f"# {TITLE}", elem_id="header")
    stats_md = gr.Markdown(value=_stats_md, elem_id="stats")

    with gr.Row(elem_id="model-row"):
        llm_dd = gr.Dropdown(
            choices=model_list(),
            value=DEFAULT_MODEL,
            label="🧠 Model",
            scale=2,
            interactive=True,
        )
        embed_dd = gr.Dropdown(
            choices=list(EMBED_MODELS.keys()),
            value=DEFAULT_EMBED,
            label="📐 Embedding",
            scale=2,
            interactive=True,
        )
        refresh_btn = gr.Button("🔄", scale=0, min_width=50)

    chatbot = gr.Chatbot(
        label="CyberTA",
        height=520,
        type="messages",
        avatar_images=(
            None,
            "https://api.dicebear.com/7.x/bottts/svg?seed=CyberTA",
        ),
        show_copy_button=True,
    )

    with gr.Row():
        msg_box  = gr.Textbox(
            placeholder="Ask a question about your course materials…",
            label="",
            scale=5,
            lines=1,
            autofocus=True,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

    clear_btn = gr.Button("🧹 Clear Chat", size="sm")

    sel_llm   = gr.State(DEFAULT_MODEL)
    sel_embed = gr.State(DEFAULT_EMBED)

    llm_dd.change(lambda v: v,   [llm_dd],   [sel_llm])
    embed_dd.change(lambda v: v, [embed_dd], [sel_embed])
    refresh_btn.click(_stats_md, outputs=[stats_md])

    send_btn.click(handle_chat,
                   [msg_box, chatbot, sel_llm, sel_embed],
                   [chatbot, msg_box])
    msg_box.submit(handle_chat,
                   [msg_box, chatbot, sel_llm, sel_embed],
                   [chatbot, msg_box])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=True,
    )

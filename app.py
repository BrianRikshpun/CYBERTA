"""
CyberTA – Gradio App (lightweight)
Assumes the VDB is already built by build_vdb.py.
All heavy imports are deferred inside functions — app starts instantly.
"""

import os
import gradio as gr
from llm_providers import model_list, provider_of, DEFAULT_MODEL, ALL_MODELS

# ── Embedding model list (no heavy import needed) ─────────────────────────────
EMBED_MODELS = {
    "MiniLM-L6-v2 (fast, 384d)"          : "all-MiniLM-L6-v2",
    "MiniLM-L12-v2 (balanced, 384d)"     : "all-MiniLM-L12-v2",
    "MPNet-base-v2 (best quality, 768d)" : "all-mpnet-base-v2",
    "BGE-small-en-v1.5 (fast, 384d)"     : "BAAI/bge-small-en-v1.5",
    "BGE-base-en-v1.5 (quality, 768d)"   : "BAAI/bge-base-en-v1.5",
    "E5-small-v2 (384d)"                 : "intfloat/e5-small-v2",
    "GTE-small (384d)"                   : "thenlper/gte-small",
}
DEFAULT_EMBED = "MiniLM-L6-v2 (fast, 384d)"

# ── Lazy pipeline import ───────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        import rag_pipeline as rp
        _pipeline = rp
    return _pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats_md():
    try:
        rp = get_pipeline()
        s  = rp.collection_stats()
        if s["total_chunks"] == 0:
            return "📭 VDB is empty. Run `python build_vdb.py` on Hamming first."
        src_list = "\n".join(f"  - {src}" for src in sorted(s["sources"]))
        return (f"📊 **{s['total_chunks']} chunks** from "
                f"**{len(s['sources'])} doc(s)**:\n{src_list}")
    except Exception as e:
        return f"⚠️ Could not load VDB: {e}"

def _provider_note(llm_name):
    p = provider_of(llm_name)
    notes = {
        "HuggingFace"   : "🤗 Needs **HF_TOKEN**. Free at huggingface.co → Settings → Access Tokens.",
        "Ollama (local)": "🖥️  Needs Ollama running. On Hamming: `/share/apps/ollama/bin/ollama serve &`",
        "OpenAI"        : "🔑 Needs **OPENAI_API_KEY**.",
        "Anthropic"     : "🔑 Needs **ANTHROPIC_API_KEY**.",
    }
    return notes.get(p, "")


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_save_keys(hf_tok, openai_key, anthropic_key, ollama_host):
    if hf_tok.strip():        os.environ["HF_TOKEN"]          = hf_tok.strip()
    if openai_key.strip():    os.environ["OPENAI_API_KEY"]     = openai_key.strip()
    if anthropic_key.strip(): os.environ["ANTHROPIC_API_KEY"]  = anthropic_key.strip()
    if ollama_host.strip():   os.environ["OLLAMA_HOST"]        = ollama_host.strip()
    return "✅ Keys saved for this session."

def handle_refresh_stats():
    return _stats_md()

def handle_visualise(method):
    try:
        rp = get_pipeline()
        from visualize import build_2d_plot, build_3d_plot
        embeddings, documents, sources = rp.get_all_vectors()
        if not embeddings:
            return None, None, "⚠️ VDB is empty. Run build_vdb.py first."
        fig2d = build_2d_plot(embeddings, documents, sources, method=method.lower())
        fig3d = build_3d_plot(embeddings, documents, sources, method=method.lower())
        return fig2d, fig3d, "✅ Plots generated."
    except Exception as e:
        return None, None, f"❌ {e}"

def handle_chat(user_msg, history, llm_name, embed_name):
    if not user_msg.strip():
        return history, ""
    try:
        rp = get_pipeline()
        s  = rp.collection_stats()
        if s["total_chunks"] == 0:
            reply   = "⚠️ VDB is empty. Run `python build_vdb.py` on Hamming first."
            sources = []
        else:
            plain_history = []
            for turn in history:
                plain_history.append({"role": "user",      "content": turn[0]})
                plain_history.append({"role": "assistant",  "content": turn[1]})
            reply, sources = rp.rag_query(
                user_msg, plain_history,
                llm_display_name   = llm_name,
                embed_display_name = embed_name,
            )
        if sources:
            reply += f"\n\n> 📎 *Sources: {', '.join(sources)}*"
    except Exception as e:
        reply = f"❌ {e}"
    history.append((user_msg, reply))
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────

TITLE = "🤖 CyberTA – AI Teaching Assistant"

with gr.Blocks(title=TITLE) as demo:

    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        "RAG-powered teaching assistant. "
        "Build the knowledge base with `python build_vdb.py`, then chat here."
    )

    sel_llm   = gr.State(DEFAULT_MODEL)
    sel_embed = gr.State(DEFAULT_EMBED)

    with gr.Tabs():

        # ── Tab 1 – Model Setup ───────────────────────────────────────────────
        with gr.TabItem("⚙️ Model Setup"):
            gr.Markdown("### Choose LLM + Embedding Model")
            with gr.Row():
                with gr.Column():
                    llm_dd = gr.Dropdown(
                        choices=model_list(), value=DEFAULT_MODEL,
                        label="🧠 Language Model (generation)",
                    )
                    provider_note = gr.Markdown(value=_provider_note(DEFAULT_MODEL))

                with gr.Column():
                    embed_dd = gr.Dropdown(
                        choices=list(EMBED_MODELS.keys()), value=DEFAULT_EMBED,
                        label="📐 Embedding Model",
                        info="Must match the model used in build_vdb.py",
                    )

            gr.Markdown("### API Keys")
            with gr.Row():
                hf_tok_box    = gr.Textbox(label="HF_TOKEN", type="password", placeholder="hf_…")
                openai_box    = gr.Textbox(label="OPENAI_API_KEY", type="password", placeholder="sk-…")
            with gr.Row():
                anthropic_box = gr.Textbox(label="ANTHROPIC_API_KEY", type="password", placeholder="sk-ant-…")
                ollama_box    = gr.Textbox(label="OLLAMA_HOST", placeholder="http://localhost:11434")

            save_btn = gr.Button("💾 Save Keys", variant="primary")
            save_msg = gr.Markdown()

            save_btn.click(handle_save_keys,
                           [hf_tok_box, openai_box, anthropic_box, ollama_box],
                           [save_msg])
            llm_dd.change(lambda v: (v, _provider_note(v)),
                          [llm_dd], [sel_llm, provider_note])
            embed_dd.change(lambda v: v, [embed_dd], [sel_embed])


        # ── Tab 2 – VDB Status ────────────────────────────────────────────────
        with gr.TabItem("📚 Knowledge Base"):
            gr.Markdown("### VDB Status")
            gr.Markdown(
                "Build the VDB by running this command on Hamming:\n"
                "```bash\n"
                "python build_vdb.py\n"
                "```"
            )
            stats_box   = gr.Markdown(value="Click Refresh to check VDB status.")
            refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            refresh_btn.click(handle_refresh_stats, outputs=[stats_box])


        # ── Tab 3 – Visualise ─────────────────────────────────────────────────
        with gr.TabItem("🔭 Visualise Vectors"):
            gr.Markdown("### Vector Space Explorer")
            gr.Markdown("Each point = one text chunk. Colour = source document.")
            with gr.Row():
                method_dd = gr.Dropdown(choices=["PCA","UMAP"], value="PCA",
                                        label="Method", scale=1)
                viz_btn   = gr.Button("🚀 Generate Plots", variant="primary", scale=2)
            viz_status = gr.Markdown()
            with gr.Row():
                plot_2d = gr.Plot(label="2D")
                plot_3d = gr.Plot(label="3D")
            viz_btn.click(handle_visualise, [method_dd], [plot_2d, plot_3d, viz_status])


        # ── Tab 4 – Chat ──────────────────────────────────────────────────────
        with gr.TabItem("💬 Ask CyberTA"):
            gr.Markdown("### Chat with Your Documents")
            active_md = gr.Markdown(
                value=f"🧠 **{DEFAULT_MODEL}**  |  📐 **{DEFAULT_EMBED}**"
            )
            chatbot = gr.Chatbot(label="CyberTA", height=460)
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="Ask a question…", scale=5, lines=1)
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)
            clear_btn = gr.Button("🧹 Clear Chat")

            send_btn.click(handle_chat,
                           [msg_box, chatbot, sel_llm, sel_embed],
                           [chatbot, msg_box])
            msg_box.submit(handle_chat,
                           [msg_box, chatbot, sel_llm, sel_embed],
                           [chatbot, msg_box])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

            sel_llm.change(
                lambda l, e: f"🧠 **{l}**  |  📐 **{e}**",
                [sel_llm, sel_embed], [active_md])
            sel_embed.change(
                lambda l, e: f"🧠 **{l}**  |  📐 **{e}**",
                [sel_llm, sel_embed], [active_md])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=True,
    )

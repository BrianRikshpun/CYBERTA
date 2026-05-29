"""
CyberTA – Gradio Application
Tabs:
  1. ⚙️  Model Setup            – choose LLM + embedding model, enter API keys
  2. 📚 Build Knowledge Base   – upload & index documents
  3. 🔭 Visualise Vectors       – 2D & 3D cluster plots
  4. 💬 Ask CyberTA             – RAG chat
"""

import os
import gradio as gr

from rag_pipeline import (
    index_file, collection_stats, get_all_vectors,
    rag_query, reset_collection,
    embed_model_list, DEFAULT_EMBED,
)
from llm_providers import model_list, provider_of, DEFAULT_MODEL, ALL_MODELS
from visualize import build_2d_plot, build_3d_plot, save_plots

# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats_md() -> str:
    s = collection_stats()
    if s["total_chunks"] == 0:
        return "📭 Knowledge base is **empty**. Upload documents to get started."
    src_list = "\n".join(f"  - {src}" for src in sorted(s["sources"]))
    return (
        f"📊 **{s['total_chunks']} chunks** from "
        f"**{len(s['sources'])} doc(s)**:\n{src_list}"
    )

def _provider_note(llm_name: str) -> str:
    p = provider_of(llm_name)
    notes = {
        "HuggingFace"   : "🤗 Needs **HF_TOKEN** (free at huggingface.co). Some gated models need a licence accept.",
        "Ollama (local)": "🖥️  Needs **Ollama** running locally. Run `ollama pull <model>` first.",
        "OpenAI"        : "🔑 Needs **OPENAI_API_KEY** (paid account).",
        "Anthropic"     : "🔑 Needs **ANTHROPIC_API_KEY**.",
    }
    return notes.get(p, "")


# ── Tab 1 – Model Setup ───────────────────────────────────────────────────────

def handle_save_keys(hf_tok, openai_key, anthropic_key, ollama_host):
    if hf_tok.strip():       os.environ["HF_TOKEN"]           = hf_tok.strip()
    if openai_key.strip():   os.environ["OPENAI_API_KEY"]      = openai_key.strip()
    if anthropic_key.strip():os.environ["ANTHROPIC_API_KEY"]   = anthropic_key.strip()
    if ollama_host.strip():  os.environ["OLLAMA_HOST"]         = ollama_host.strip()
    return "✅ Keys saved to environment for this session."

def on_llm_change(llm_name):
    return _provider_note(llm_name)


# ── Tab 2 – Upload & Index ────────────────────────────────────────────────────

def handle_upload(files, embed_model):
    if not files:
        return "⚠️ No files selected.", _stats_md()
    log_lines = []
    def cb(msg): log_lines.append(msg)
    for f in files:
        try:
            index_file(f.name, embed_model_name=embed_model, progress_cb=cb)
        except Exception as e:
            log_lines.append(f"❌ **{os.path.basename(f.name)}**: {e}")
    return "\n\n".join(log_lines), _stats_md()

def handle_reset():
    try:
        reset_collection()
        return "🗑️ Knowledge base cleared.", _stats_md()
    except Exception as e:
        return f"❌ {e}", _stats_md()


# ── Tab 3 – Visualise ─────────────────────────────────────────────────────────

def handle_visualise(method):
    embeddings, documents, sources = get_all_vectors()
    if not embeddings:
        return None, None, "⚠️ No vectors yet. Index some documents first."
    fig2d = build_2d_plot(embeddings, documents, sources, method=method.lower())
    fig3d = build_3d_plot(embeddings, documents, sources, method=method.lower())
    p2d, p3d = save_plots(fig2d, fig3d)
    return fig2d, fig3d, f"✅ Saved → `{p2d}` & `{p3d}`"


# ── Tab 4 – RAG Chat ──────────────────────────────────────────────────────────

def handle_chat(user_msg, history, llm_name, embed_name):
    if not user_msg.strip():
        return history, ""

    plain_history = []
    for turn in history:
        plain_history.append({"role": "user",      "content": turn[0]})
        plain_history.append({"role": "assistant",  "content": turn[1]})

    s = collection_stats()
    if s["total_chunks"] == 0:
        reply = "⚠️ Knowledge base empty — please upload documents first."
        sources = []
    else:
        try:
            reply, sources = rag_query(
                user_msg, plain_history,
                llm_display_name=llm_name,
                embed_display_name=embed_name,
            )
        except Exception as e:
            reply, sources = f"❌ {e}", []

    if sources:
        reply += f"\n\n> 📎 *Sources: {', '.join(sources)}*"

    history.append((user_msg, reply))
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────

TITLE = "🤖 CyberTA – AI Teaching Assistant"

CSS = """
body, .gradio-container { font-family: 'Inter', sans-serif; }
.tab-nav button         { font-size: 1rem; font-weight: 600; }
#stats-box              { background:#1a1a2e; border-radius:8px; padding:12px; color:#a0c4ff; }
#log-box                { font-family:monospace; font-size:.85rem; }
#provider-note          { background:#1e2a1e; border-radius:6px; padding:8px;
                          color:#90ee90; font-size:.9rem; min-height:36px; }
footer                  { display:none !important; }
"""

with gr.Blocks(title=TITLE, theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:

    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        "Upload course materials → explore vector space → chat with an AI "
        "that answers **only from your documents** using a fully open-source stack."
    )

    # ── Shared state: selected LLM & embedding model ──────────────────────────
    sel_llm   = gr.State(DEFAULT_MODEL)
    sel_embed = gr.State(DEFAULT_EMBED)

    with gr.Tabs():

        # ── Tab 1 – Model Setup ───────────────────────────────────────────────
        with gr.TabItem("⚙️ Model Setup"):
            gr.Markdown("### Choose your LLM and Embedding model")

            with gr.Row():
                with gr.Column():
                    llm_dd = gr.Dropdown(
                        choices=model_list(),
                        value=DEFAULT_MODEL,
                        label="🧠 Language Model (generation)",
                        info="Pick a model. Provider requirements shown below.",
                    )
                    provider_note = gr.Markdown(
                        value=_provider_note(DEFAULT_MODEL),
                        elem_id="provider-note",
                    )

                with gr.Column():
                    embed_dd = gr.Dropdown(
                        choices=embed_model_list(),
                        value=DEFAULT_EMBED,
                        label="📐 Embedding Model (indexing + retrieval)",
                        info="All run locally via sentence-transformers. No API key needed.",
                    )
                    gr.Markdown(
                        "💡 Re-index documents if you change the embedding model — "
                        "vectors must match the model used at index time.",
                        elem_id="provider-note",
                    )

            gr.Markdown("---")
            gr.Markdown("### API Keys & Endpoints")
            gr.Markdown(
                "Keys are stored **only in memory** for this session. "
                "On Hugging Face Spaces, set them as **Secrets** (not here)."
            )

            with gr.Row():
                hf_tok_box    = gr.Textbox(label="HF_TOKEN (HuggingFace)",
                                           type="password", placeholder="hf_…")
                openai_box    = gr.Textbox(label="OPENAI_API_KEY",
                                           type="password", placeholder="sk-…")
            with gr.Row():
                anthropic_box = gr.Textbox(label="ANTHROPIC_API_KEY",
                                           type="password", placeholder="sk-ant-…")
                ollama_box    = gr.Textbox(label="OLLAMA_HOST (default: http://localhost:11434)",
                                           placeholder="http://localhost:11434")

            save_btn  = gr.Button("💾 Save Keys", variant="primary")
            save_msg  = gr.Markdown()

            save_btn.click(
                handle_save_keys,
                inputs=[hf_tok_box, openai_box, anthropic_box, ollama_box],
                outputs=[save_msg],
            )
            llm_dd.change(on_llm_change, inputs=[llm_dd], outputs=[provider_note])
            llm_dd.change(lambda v: v,   inputs=[llm_dd],   outputs=[sel_llm])
            embed_dd.change(lambda v: v, inputs=[embed_dd], outputs=[sel_embed])


        # ── Tab 2 – Build Knowledge Base ──────────────────────────────────────
        with gr.TabItem("📚 Build Knowledge Base"):
            gr.Markdown("### Upload & Index Documents")
            gr.Markdown("Supported: **PDF · DOCX · PPTX**. Multiple files at once.")

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Drop files here",
                        file_types=[".pdf",".docx",".doc",".pptx",".ppt"],
                        file_count="multiple",
                    )
                    embed_info = gr.Markdown(
                        value=lambda: f"Using embedding: **{DEFAULT_EMBED}** *(change in Model Setup)*"
                    )
                    with gr.Row():
                        index_btn = gr.Button("🔨 Index Documents", variant="primary")
                        reset_btn = gr.Button("🗑️ Clear KB", variant="stop")

                with gr.Column(scale=1):
                    stats_box = gr.Markdown(value=_stats_md, elem_id="stats-box")

            log_box = gr.Markdown(elem_id="log-box")

            index_btn.click(handle_upload, [file_input, sel_embed], [log_box, stats_box])
            reset_btn.click(handle_reset,  [],                       [log_box, stats_box])
            sel_embed.change(
                lambda v: f"Using embedding: **{v}**",
                inputs=[sel_embed], outputs=[embed_info],
            )


        # ── Tab 3 – Visualise Vectors ─────────────────────────────────────────
        with gr.TabItem("🔭 Visualise Vectors"):
            gr.Markdown("### Explore Vector Space")
            gr.Markdown(
                "Each point = one text chunk. **Colour = source document.** "
                "Nearby points share similar meaning."
            )

            with gr.Row():
                method_dd = gr.Dropdown(
                    choices=["PCA","UMAP"], value="PCA",
                    label="Dimensionality Reduction", scale=1,
                )
                viz_btn = gr.Button("🚀 Generate Plots", variant="primary", scale=2)

            viz_status = gr.Markdown()

            with gr.Row():
                plot_2d = gr.Plot(label="2D Cluster View")
                plot_3d = gr.Plot(label="3D Cluster View")

            viz_btn.click(handle_visualise, [method_dd], [plot_2d, plot_3d, viz_status])


        # ── Tab 4 – RAG Chat ──────────────────────────────────────────────────
        with gr.TabItem("💬 Ask CyberTA"):
            gr.Markdown("### Chat with Your Documents")

            with gr.Row():
                active_llm_md   = gr.Markdown(
                    value=lambda: f"🧠 LLM: **{DEFAULT_MODEL}**   |   "
                                  f"📐 Embed: **{DEFAULT_EMBED}**"
                )

            chatbot = gr.Chatbot(
                label="CyberTA",
                height=460,
                bubble_full_width=False,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=CyberTA",
                ),
            )

            with gr.Row():
                msg_box  = gr.Textbox(
                    placeholder="Ask a question about your course materials…",
                    label="Your question", scale=5, lines=1,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

            clear_btn = gr.Button("🧹 Clear Chat")

            send_btn.click(handle_chat,
                           [msg_box, chatbot, sel_llm, sel_embed],
                           [chatbot, msg_box])
            msg_box.submit(handle_chat,
                           [msg_box, chatbot, sel_llm, sel_embed],
                           [chatbot, msg_box])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

            # Keep active-model badge in sync
            sel_llm.change(
                lambda l, e: f"🧠 LLM: **{l}**   |   📐 Embed: **{e}**",
                [sel_llm, sel_embed], [active_llm_md],
            )
            sel_embed.change(
                lambda l, e: f"🧠 LLM: **{l}**   |   📐 Embed: **{e}**",
                [sel_llm, sel_embed], [active_llm_md],
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )

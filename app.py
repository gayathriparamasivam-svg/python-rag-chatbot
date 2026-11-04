import hashlib
import html
import math
import re
from io import BytesIO

import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DocLens · Chat with your PDF",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"
MIN_SCORE = 0.25

ANSWER_STYLES = {
    "Concise": "Answer in 2-4 short sentences.",
    "Detailed": "Give a thorough, well-structured answer using short paragraphs or bullet points.",
    "Explain simply": "Explain in plain words for a beginner, with a short example if helpful.",
}

# ----------------------------------------------------------------------
# Visual design: midnight aurora, glass panels, glowing page map
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=Manrope:wght@400;500;700&display=swap');

:root {
  --bg: #0b1020;
  --panel: rgba(255,255,255,0.055);
  --panel-line: rgba(255,255,255,0.12);
  --text: #e8ecff;
  --muted: #94a0cf;
  --violet: #7c5cff;
  --cyan: #22d3ee;
  --pink: #ff5caa;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }

html, body, .stApp, .stMarkdown, p, li, label, span, div { font-family: 'Manrope', sans-serif; }
h1, h2, h3, h4, .dl-title, .dl-num { font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }

/* Aurora background */
.stApp {
  background:
    radial-gradient(900px 500px at 12% -10%, rgba(124,92,255,0.38), transparent 60%),
    radial-gradient(800px 500px at 95% 5%, rgba(34,211,238,0.22), transparent 60%),
    radial-gradient(700px 500px at 60% 110%, rgba(255,92,170,0.18), transparent 60%),
    var(--bg);
}
.block-container { padding-top: 2.2rem; max-width: 1050px; }

/* Hero */
.dl-hero { margin: 4px 0 26px 0; }
.dl-badge { display: inline-block; padding: 5px 14px; border-radius: 99px; font-size: 0.82rem; font-weight: 700;
  color: var(--cyan); background: rgba(34,211,238,0.1); border: 1px solid rgba(34,211,238,0.35); margin-bottom: 16px; }
.dl-title { font-size: clamp(2.2rem, 5vw, 3.7rem); font-weight: 800; line-height: 1.04; margin: 0;
  background: linear-gradient(100deg, #ffffff 0%, #b7a8ff 40%, #22d3ee 75%, #ff5caa 100%);
  background-size: 220% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 9s ease-in-out infinite alternate; }
@keyframes shimmer { from { background-position: 0% 0; } to { background-position: 100% 0; } }
.dl-sub { color: var(--muted); font-size: 1.08rem; margin-top: 12px; max-width: 52ch; line-height: 1.55; }

/* Glass panels */
.dl-glass { background: var(--panel); border: 1px solid var(--panel-line); border-radius: 20px;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }

/* Stat cards */
.dl-stats { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 12px; margin-bottom: 14px; }
.dl-stat { padding: 14px 18px; }
.dl-stat .dl-num { font-size: 1.7rem; font-weight: 800; color: #fff; line-height: 1.1; }
.dl-stat .dl-lab { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
.dl-stat .dl-file { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem; color: #fff;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
@media (max-width: 700px) { .dl-stats { grid-template-columns: 1fr 1fr; } }

/* Page map */
.dl-map { padding: 16px 18px; margin-bottom: 18px; }
.dl-map-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.85rem; margin-bottom: 10px; }
.dl-tiles { display: flex; flex-wrap: wrap; gap: 5px; }
.dl-tile { min-width: 22px; height: 28px; padding: 0 4px; border-radius: 6px; font-size: 0.68rem; display: flex;
  align-items: center; justify-content: center; color: rgba(255,255,255,0.35);
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); }
.dl-tile.hit { color: #0b1020; font-weight: 800; border-color: transparent;
  background: linear-gradient(135deg, var(--cyan), var(--violet));
  box-shadow: 0 0 14px rgba(34,211,238,0.65); animation: pop .5s ease-out; }
@keyframes pop { 0% { transform: scale(0.6); } 70% { transform: scale(1.18); } 100% { transform: scale(1); } }

/* Empty-state feature row */
.dl-features { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 22px; }
.dl-feature { padding: 20px; }
.dl-feature h4 { margin: 0 0 6px 0; color: #fff; font-size: 1.05rem; }
.dl-feature p { margin: 0; color: var(--muted); font-size: 0.93rem; line-height: 1.5; }

/* Evidence cards */
.dl-ev { display: flex; gap: 14px; padding: 14px 16px; margin: 10px 0; border-radius: 14px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--panel-line); border-left: 4px solid var(--violet); }
.dl-ev-body { font-size: 0.93rem; line-height: 1.6; color: #d6dcf7; }
.dl-ev-body mark { background: rgba(34,211,238,0.22); color: #fff; border-bottom: 2px solid var(--cyan);
  padding: 1px 3px; border-radius: 3px; }
.dl-ev-page { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 0.85rem; color: var(--cyan); margin-bottom: 4px; }
.dl-ring { flex: 0 0 46px; height: 46px; border-radius: 50%; display: grid; place-items: center; font-size: 0.72rem;
  font-weight: 800; color: #fff;
  background: conic-gradient(var(--cyan) calc(var(--p) * 1%), rgba(255,255,255,0.12) 0); position: relative; }
.dl-ring::before { content: ""; position: absolute; inset: 4px; border-radius: 50%; background: #151c3a; }
.dl-ring span { position: relative; }

/* Chat messages */
[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--panel-line);
  border-radius: 18px; padding: 16px 18px; backdrop-filter: blur(10px); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, rgba(124,92,255,0.28), rgba(34,211,238,0.12)); border-color: rgba(124,92,255,0.45); }

/* Chat input */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] { border-radius: 18px; border: 1px solid rgba(124,92,255,0.55);
  background: rgba(20,27,54,0.9); box-shadow: 0 0 28px rgba(124,92,255,0.25); }

/* Buttons and chips */
.stButton > button { border-radius: 99px; border: 1px solid rgba(124,92,255,0.6); background: rgba(124,92,255,0.12);
  color: var(--text); font-weight: 500; padding: 6px 18px; transition: all .15s ease; }
.stButton > button:hover { background: linear-gradient(100deg, var(--violet), var(--cyan)); color: #0b1020; border-color: transparent; }
.stDownloadButton > button { border-radius: 99px; }

/* Uploader */
[data-testid="stFileUploader"] section { background: var(--panel); border: 2px dashed rgba(34,211,238,0.5);
  border-radius: 20px; padding: 30px; }
[data-testid="stFileUploader"] section:hover { border-color: var(--pink); }

/* Expanders and sidebar */
[data-testid="stExpander"] details { background: transparent; border: 1px solid var(--panel-line); border-radius: 14px; }
[data-testid="stSidebar"] { background: rgba(11,16,32,0.92); border-right: 1px solid var(--panel-line); }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = ""

if not api_key:
    st.error("Gemini API key is missing. Add GEMINI_API_KEY to your Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# ----------------------------------------------------------------------
# RAG helpers
# ----------------------------------------------------------------------
def split_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(reader):
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in split_text(page.extract_text() or ""):
            records.append({"page": page_number, "text": chunk})
    return records


def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    vectors = []
    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
        )
        vectors.extend([item.values for item in response.embeddings])
    return np.array(vectors, dtype=np.float32)


def retrieve(question, records, embeddings, top_k):
    query = get_embeddings([question], task_type="RETRIEVAL_QUERY")[0]
    sims = (embeddings @ query) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query) + 1e-10
    )
    best = np.argsort(sims)[::-1][:top_k]
    return [{**records[i], "score": float(sims[i])} for i in best if sims[i] > MIN_SCORE]


def highlight_best_sentence(text, question):
    words = set(re.findall(r"[a-zA-Z0-9]{4,}", question.lower()))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scores = [len(words & set(re.findall(r"[a-zA-Z0-9]{4,}", s.lower()))) for s in sentences]
    best = int(np.argmax(scores)) if scores and max(scores) > 0 else -1
    return " ".join(
        f"<mark>{html.escape(s)}</mark>" if i == best else html.escape(s)
        for i, s in enumerate(sentences)
    )


def stream_answer(question, sources, history, style):
    context = "\n\n".join(f"[Page {s['page']}] {s['text']}" for s in sources)
    past = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history[-6:])
    system = (
        "You answer questions about a PDF using only the supplied context. "
        "If the answer is not in the context, say you cannot find it in the document. "
        "Mention page numbers like (p. 3) when you rely on a passage. "
        f"{ANSWER_STYLES[style]}"
    )
    prompt = f"Conversation so far:\n{past}\n\nContext:\n{context}\n\nQuestion: {question}"
    for piece in client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.3),
    ):
        if piece.text:
            yield piece.text


def suggest_questions(records):
    sample = "\n".join(r["text"] for r in records[:6])[:4000]
    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=(
                "Write exactly 3 short, specific questions a reader could ask about this document. "
                "One question per line, no numbering, no extra text.\n\n" + sample
            ),
        )
        lines = [l.strip(" -•\t") for l in (response.text or "").splitlines() if l.strip()]
        return lines[:3]
    except Exception:
        return []


def render_page_map(slot, total, active):
    """Grid of page tiles. Pages used in the latest answer glow."""
    group = max(1, math.ceil(total / 60))
    tiles = []
    for start in range(1, total + 1, group):
        end = min(total, start + group - 1)
        hit = any(p in active for p in range(start, end + 1))
        label = str(start) if group == 1 else ""
        title = f"Page {start}" if group == 1 else f"Pages {start}-{end}"
        tiles.append(f'<div class="dl-tile {"hit" if hit else ""}" title="{title}">{label}</div>')
    note = (
        "Glowing pages were used for the latest answer"
        if active
        else "Ask a question and the pages it came from will light up"
    )
    slot.markdown(
        f'<div class="dl-map dl-glass"><div class="dl-map-head"><b>Page map</b><span>{note}</span></div>'
        f'<div class="dl-tiles">{"".join(tiles)}</div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources, question):
    for s in sources:
        pct = max(0, min(100, int(s["score"] * 100)))
        st.markdown(
            f"""
<div class="dl-ev">
  <div class="dl-ring" style="--p:{pct}"><span>{pct}</span></div>
  <div><div class="dl-ev-page">Page {s['page']}</div>
  <div class="dl-ev-body">{highlight_best_sentence(s['text'], question)}</div></div>
</div>""",
            unsafe_allow_html=True,
        )


def chat_to_markdown(messages, name):
    lines = [f"# Chat about {name}", ""]
    for m in messages:
        who = "You" if m["role"] == "user" else "DocLens"
        lines += [f"**{who}:** {m['content']}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔮 DocLens")
    style = st.radio("Answer style", list(ANSWER_STYLES), index=0)
    top_k = st.slider("Passages per question", 2, 8, 4)

    if "records" in st.session_state:
        st.markdown("---")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.active_pages = set()
            st.rerun()
        if st.session_state.get("messages"):
            st.download_button(
                "Save chat as Markdown",
                data=chat_to_markdown(st.session_state.messages, st.session_state.pdf_name),
                file_name="chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    """
<div class="dl-hero">
  <div class="dl-badge">Retrieval-augmented answers with sources</div>
  <h1 class="dl-title">Ask your PDF anything.<br>See exactly where the answer came from.</h1>
  <div class="dl-sub">Upload a document, ask in plain language, and watch the pages behind each answer light up.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Upload (large in the hero when empty, tucked away once a PDF is loaded)
# ----------------------------------------------------------------------
if "records" in st.session_state:
    with st.expander("Switch document"):
        uploaded_file = st.file_uploader("Choose another PDF", type=["pdf"], key="pdf")
else:
    uploaded_file = st.file_uploader("Drop a PDF here to begin", type=["pdf"], key="pdf")

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        try:
            with st.status("Reading your PDF…", expanded=True) as status:
                st.write("Extracting text page by page")
                reader = PdfReader(BytesIO(file_bytes))
                records = build_chunks(reader)

                if not records:
                    status.update(label="No readable text found", state="error")
                    st.error("This looks like a scanned PDF. Upload a text-based PDF instead.")
                    st.stop()

                st.write(f"Embedding {len(records)} passages")
                embeddings = get_embeddings([r["text"] for r in records])

                st.write("Preparing starter questions")
                suggestions = suggest_questions(records)

                st.session_state.update(
                    file_hash=file_hash,
                    records=records,
                    embeddings=embeddings,
                    pdf_name=uploaded_file.name,
                    page_count=len(reader.pages),
                    suggestions=suggestions,
                    messages=[],
                    active_pages=set(),
                )
                status.update(label="Ready to chat", state="complete", expanded=False)
            st.rerun()
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
            st.stop()

# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------
if "records" not in st.session_state:
    st.markdown(
        """
<div class="dl-features">
  <div class="dl-feature dl-glass"><h4>Answers with proof</h4><p>Every reply lists the passages it used, with the key sentence highlighted.</p></div>
  <div class="dl-feature dl-glass"><h4>Page map</h4><p>A visual strip of your document where the cited pages glow after each answer.</p></div>
  <div class="dl-feature dl-glass"><h4>Starter questions</h4><p>Not sure what to ask? Get three ideas generated from your document.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------
# Document dashboard
# ----------------------------------------------------------------------
questions_asked = sum(1 for m in st.session_state.messages if m["role"] == "user")
st.markdown(
    f"""
<div class="dl-stats">
  <div class="dl-stat dl-glass"><div class="dl-file">{html.escape(st.session_state.pdf_name)}</div><div class="dl-lab">Current document</div></div>
  <div class="dl-stat dl-glass"><div class="dl-num">{st.session_state.page_count}</div><div class="dl-lab">Pages</div></div>
  <div class="dl-stat dl-glass"><div class="dl-num">{len(st.session_state.records)}</div><div class="dl-lab">Passages indexed</div></div>
  <div class="dl-stat dl-glass"><div class="dl-num">{questions_asked}</div><div class="dl-lab">Questions asked</div></div>
</div>
""",
    unsafe_allow_html=True,
)

map_slot = st.empty()
render_page_map(map_slot, st.session_state.page_count, st.session_state.get("active_pages", set()))

# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Evidence · {len(message['sources'])} passages"):
                render_sources(message["sources"], message.get("question", ""))

pending = st.session_state.pop("pending_question", None)
if not st.session_state.messages and st.session_state.get("suggestions"):
    st.caption("Try asking")
    for n, suggestion in enumerate(st.session_state.suggestions):
        if st.button(suggestion, key=f"sugg_{n}"):
            st.session_state.pending_question = suggestion
            st.rerun()

typed = st.chat_input("Ask something about your PDF…")
question = typed or pending

if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding the right pages…"):
                sources = retrieve(
                    question,
                    st.session_state.records,
                    st.session_state.embeddings,
                    top_k,
                )

            if not sources:
                answer = "I could not find relevant information in the uploaded document."
                st.markdown(answer)
            else:
                answer = st.write_stream(stream_answer(question, sources, history, style))
                with st.expander(f"Evidence · {len(sources)} passages"):
                    render_sources(sources, question)

            st.session_state.active_pages = {s["page"] for s in sources}
            render_page_map(map_slot, st.session_state.page_count, st.session_state.active_pages)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "question": question}
            )
        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

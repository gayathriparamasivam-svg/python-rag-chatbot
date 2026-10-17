
import streamlit as st
import numpy as np
from io import BytesIO
from pypdf import PdfReader
from google import genai

st.set_page_config(
    page_title="Python RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Python RAG Chatbot")
st.write("Upload a PDF and ask questions about its contents.")

# Read API key from Streamlit secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = ""

if not api_key:
    st.error(
        "Gemini API key is missing. "
        "Add GEMINI_API_KEY to your Streamlit secrets."
    )
    st.stop()

client = genai.Client(api_key=api_key)

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"


def split_text(text, chunk_size=1000, overlap=150):
    """Split document text into overlapping chunks."""
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def get_embeddings(texts):
    """Create embeddings in small batches."""
    vectors = []

    for start in range(0, len(texts), 20):
        batch = texts[start:start + 20]

        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config={"output_dimensionality": 768}
        )

        vectors.extend(
            [item.values for item in response.embeddings]
        )

    return np.array(vectors, dtype=np.float32)


def retrieve_chunks(question, chunks, embeddings, top_k=4):
    """Retrieve the most relevant document chunks."""
    query_vector = get_embeddings([question])[0]

    # Cosine similarity
    chunk_norms = np.linalg.norm(
        embeddings, axis=1
    )
    query_norm = np.linalg.norm(query_vector)

    similarities = (
        embeddings @ query_vector
    ) / (
        chunk_norms * query_norm + 1e-10
    )

    best_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        chunks[i]
        for i in best_indices
        if similarities[i] > 0
    ]


# Upload a PDF
uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"]
)

if uploaded_file:
    if st.button("📚 Process PDF"):
        try:
            with st.spinner("Reading and indexing your PDF..."):
                reader = PdfReader(
                    BytesIO(uploaded_file.getvalue())
                )

                pages = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]
                document_text = "\n".join(pages)

                if not document_text.strip():
                    st.error(
                        "No readable text found. "
                        "Try a text-based PDF instead of a scanned PDF."
                    )
                    st.stop()

                chunks = split_text(document_text)

                embeddings = get_embeddings(chunks)

                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            st.success(
                f"PDF processed successfully! "
                f"{len(chunks)} text chunks indexed."
            )

        except Exception as error:
            st.error(f"PDF processing failed: {error}")

# Chat interface
if "chunks" in st.session_state:
    st.subheader(
        f"Ask questions about: {st.session_state.pdf_name}"
    )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("Searching your document..."):
                relevant_chunks = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings
                )

                if not relevant_chunks:
                    answer = (
                        "I could not find relevant information "
                        "in the uploaded document."
                    )
                else:
                    context = "\n\n".join(relevant_chunks)

                    response = client.models.generate_content(
                        model=CHAT_MODEL,
                        contents=(
                            "Answer the user's question using "
                            "only the context below. If the answer "
                            "is not in the context, say you cannot "
                            "find it in the document.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        )
                    )

                    answer = response.text or (
                        "The AI did not return a text answer."
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as error:
            st.error(f"Could not generate an answer: {error}")

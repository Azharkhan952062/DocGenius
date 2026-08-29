import os
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import google.generativeai as genai

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# -----------------------------
# Page Configuration
# -----------------------------
img = Image.open(
    "Artifacts/Image Resources/images.jpeg"
)

st.set_page_config(
    page_title="DocGenius: Document Generation AI",
    page_icon=img,
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------
# st.markdown(
#     """
#     <style>

#     .stApp {
#         background-color: #0e1117;
#     }

#     .main-title {
#         text-align: center;
#         font-size: 42px;
#         font-weight: 700;
#         margin-top: 20px;
#         margin-bottom: 35px;
#         color: white;
#     }

#     .subtitle {
#         text-align: center;
#         color: #b8b8b8;
#         margin-bottom: 30px;
#     }

#     .answer-box {
#         background-color: #123d25;
#         border-radius: 6px;
#         padding: 18px;
#         color: #d8ffe5;
#         margin-top: 20px;
#     }

#     .footer {
#         text-align: center;
#         color: #777;
#         margin-top: 50px;
#         font-size: 13px;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )
# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>

    /* =========================
       MAIN PROJECT BACKGROUND
       ========================= */
    .stApp {
        background-color: #071c28;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    /* =========================
       MAIN TITLE
       ========================= */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }


    /* =========================
       SUBTITLE
       ========================= */
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 35px;
    }


    /* =========================
       PDF UPLOAD BOX
       ========================= */
    [data-testid="stFileUploader"] {
        background-color: #102b3a;
        border: 2px dashed #00e5ff;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }


    /* =========================
       UPLOADER TEXT
       ========================= */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #102b3a;
    }


    /* =========================
       SUCCESS / STATUS BOX
       ========================= */
    [data-testid="stAlert"] {
        border-radius: 12px;
        font-size: 16px;
    }


    /* =========================
       QUESTION INPUT
       ========================= */
    [data-testid="stTextInput"] input {
        background-color: #102b3a;
        color: #ffffff;
        border: 2px solid #3b5565;
        border-radius: 12px;
        padding: 14px;
        font-size: 16px;
    }


    /* =========================
       QUESTION INPUT - FOCUS
       ========================= */
    [data-testid="stTextInput"] input:focus {
        border-color: #00e5ff;
        box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.15);
    }


    /* =========================
       INPUT PLACEHOLDER
       ========================= */
    [data-testid="stTextInput"] input::placeholder {
        color: #94a3b8;
    }


    /* =========================
       LABELS
       ========================= */
    label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }


    /* =========================
       EXPANDER
       ========================= */
    [data-testid="stExpander"] {
        background-color: #102b3a;
        border-radius: 12px;
        border: 1px solid #3b5565;
    }


    /* =========================
       EXPANDER TEXT
       ========================= */
    [data-testid="stExpander"] p {
        color: #e2e8f0;
    }


    /* =========================
       NORMAL TEXT
       ========================= */
    .stMarkdown,
    .stText {
        color: #e2e8f0;
    }


    /* =========================
       FOOTER
       ========================= */
    .footer {
        text-align: center;
        color: #718797;
        margin-top: 45px;
        font-size: 13px;
    }


    /* =========================
       SCROLLBAR
       ========================= */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #071c28;
    }

    ::-webkit-scrollbar-thumb {
        background: #294655;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #00e5ff;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">Ask Your PDF 📄</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions about its content.</div>',
    unsafe_allow_html=True
)


# -----------------------------
# PDF Upload
# -----------------------------
pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if pdf is not None:

    # -------------------------
    # Read PDF
    # -------------------------
    pdf_reader = PdfReader(pdf)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    if not text.strip():

        st.error("Could not extract text from this PDF.")

    else:

        st.success("PDF uploaded successfully!")

        st.write(
            f"Characters extracted: {len(text)}"
        )


        # -------------------------
        # Text Chunking
        # -------------------------
        chunk_size = 1000
        chunk_overlap = 200

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - chunk_overlap


        st.success(
            f"Document divided into {len(chunks)} chunks."
        )


        # -------------------------
        # Gemini Embeddings
        # -------------------------
        if not GEMINI_API_KEY:

            st.error(
                "GEMINI_API_KEY is not configured."
            )

        else:

            with st.spinner(
                "Creating document embeddings..."
            ):

                embeddings = []

                for chunk in chunks:

                    result = genai.embed_content(
                      model="models/gemini-embedding-001",
                        content=chunk,
                        task_type="retrieval_document"
                    )

                    embeddings.append(
                        result["embedding"]
                    )


                embeddings = np.array(
                    embeddings,
                    dtype="float32"
                )


            # -------------------------
            # FAISS Vector Database
            # -------------------------
            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

            st.success(
                f"FAISS vector database created with {len(chunks)} chunks."
            )


            # -------------------------
            # Question
            # -------------------------
            query = st.text_input(
                "Ask your Question about your PDF"
            )


            if query:

                with st.spinner(
                    "Searching your document..."
                ):

                    # Create query embedding
                    query_result = genai.embed_content(
                        model="models/gemini-embedding-001",
                        content=query,
                        task_type="retrieval_query"
                    )

                    query_embedding = np.array(
                        [query_result["embedding"]],
                        dtype="float32"
                    )


                    # Search FAISS
                    distances, indices = index.search(
                        query_embedding,
                        min(3, len(chunks))
                    )


                    relevant_chunks = []

                    for i in indices[0]:

                        relevant_chunks.append(
                            chunks[i]
                        )


                # -------------------------
                # Generate Answer
                # -------------------------
                context = "\n\n".join(
                    relevant_chunks
                )


                prompt = f"""
You are DocGenius, a document question-answering assistant.

Answer the user's question ONLY using the information
contained in the provided document context.

If the answer is not present in the document,
say that the information is not available in the document.

Document Context:
{context}

User Question:
{query}

Give a clear and concise answer.
"""


                model = genai.GenerativeModel(
                    "gemini-3.6-flash"
                )


                response = model.generate_content(
                    prompt
                )


                # -------------------------
                # Display Answer
                # -------------------------
                st.success(
                    response.text
                )


                # -------------------------
                # Retrieved Chunks
                # -------------------------
                with st.expander(
                    "View retrieved document chunks"
                ):

                    for i, chunk in enumerate(
                        relevant_chunks
                    ):

                        st.write(
                            f"Chunk {i + 1}"
                        )

                        st.write(chunk)

                        st.divider()


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">Made with Streamlit</div>',
    unsafe_allow_html=True
)
import os
import requests
import streamlit as st
import faiss

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from requests.exceptions import Timeout, RequestException

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
    }

    .subtitle {
        text-align: center;
    }

    .question-label {
    text-align: center;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 8px;
}

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "Groq API key is missing. "
        "Please add GROQ_API_KEY to your .env file."
    )
    st.stop()

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


# -----------------------------------
# 2. Streamlit interface
# -----------------------------------

st.markdown(
    '<h1 class="main-title">📚 AI Document Q&A</h1>',
    unsafe_allow_html=True
)


st.markdown(
    '<p class="subtitle">Upload one or more PDF documents and ask questions based only on their content.</p>',
    unsafe_allow_html=True
)

st.sidebar.title("📚 Documents")

uploaded_files = st.sidebar.file_uploader(
    "📄 Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

st.markdown(
    '<div class="question-label">💬 Ask a question about your documents</div>',
    unsafe_allow_html=True
)

with st.form("question_form"):

    question = st.text_input(
        "Question",
        placeholder="e.g. What is a Transformer?",
        label_visibility="collapsed"
    )

    submitted = st.form_submit_button(
        "🔍 Ask Question"
    )

# -----------------------------------
# 3. Load embedding model
# -----------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
    # 4. Create chunks
# -----------------------------------
        
def chunk_text(text, chunk_size=500, overlap=100):
        
    chunks = []
        
    start = 0
        
    while start < len(text):
        
        end = start + chunk_size
        
        chunk = text[start:end]
        
        chunks.append(chunk)
        
        start += chunk_size - overlap
        
        return chunks

# -----------------------------------
# 5. Process PDF
# -----------------------------------

if uploaded_files:

    try:
        all_chunks = []

        st.sidebar.write(f"✓ 📚 {len(uploaded_files)} document(s) uploaded")

        for uploaded_file in uploaded_files:

            reader = PdfReader(uploaded_file)

            pdf_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pdf_text += text + "\n"

            if pdf_text.strip():

                pdf_chunks = chunk_text(pdf_text)

                all_chunks.extend(pdf_chunks)



    except Exception:
        st.error(
            "Unable to read this PDF. "
            "Please upload a valid PDF file."
        )
        st.stop()

    chunks = all_chunks

    # -----------------------------------
    # 6. Create embeddings
    # -----------------------------------

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]


    # -----------------------------------
    # 7. Create FAISS index
    # -----------------------------------

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # -----------------------------------
    # 8. Retrieve relevant chunks
    # -----------------------------------

    if submitted and question:

            question_embedding = model.encode(
                [question]
            )

            k = 3

            distances, indices = index.search(
                question_embedding.astype("float32"),
                k
            )

            # -----------------------------------
                # Choosing Relevant context
            # -----------------------------------

            best_distance = distances[0][0]

            if best_distance > 1.2:

                st.warning("This question does not appear to be covered by the uploaded PDF.")

            else:
                
                # -----------------------------------
                # 9. Create context
                # -----------------------------------

                context = ""

                for i in indices[0]:

                    context += chunks[i] + "\n\n"


                # -----------------------------------
                # 10. Create prompt
                # -----------------------------------

                prompt = f"""
                Answer the question using only the provided context.

                Context:
                {context}

                Question:
                {question}
                """


                # -----------------------------------
                # 11. Create API request body
                # -----------------------------------

                data = {

                    "model": "openai/gpt-oss-120b",

                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }


                # -----------------------------------
                # 12. Send request to AI
                # -----------------------------------
                try:    
                    response = requests.post(

                        url,

                        headers=headers,

                        json=data,

                        timeout=30
                    )

                    response.raise_for_status()

                except Timeout:
                    st.error(
                        "The AI service took too long to respond. "
                        "Please try again."
                    )
                    st.stop()

                except RequestException:
                    st.error(
                        "Unable to connect to the AI service. "
                        "Please try again later."
                    )
                    st.stop()

                # -----------------------------------
                # 13. Check API response
                # -----------------------------------

                # st.write("Status code:", response.status_code)
                # st.write("API response:", response.text)


                # -----------------------------------
                # 14. Convert JSON response
                # -----------------------------------

                result = response.json()

                answer = result["choices"][0]["message"]["content"]

                with st.container(border=True):

                    st.markdown("### 🤖 Answer")

                    st.markdown(answer)

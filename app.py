import os
import requests
import streamlit as st
import faiss

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


# -----------------------------------
# 2. Streamlit interface
# -----------------------------------

st.title("AI PDF Question Answering System")

st.write("Upload a PDF and ask questions from it.")

st.write("API key loaded:", GROQ_API_KEY is not None)


uploaded_file = st.file_uploader(
    "Upload Your PDF",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question about the PDF"
)


# -----------------------------------
# 3. Load embedding model
# -----------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------
# 4. Process PDF
# -----------------------------------

if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        all_text += text


    # -----------------------------------
    # 5. Create chunks
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


    chunks = chunk_text(all_text)


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

    st.write(
        "FAISS dimension:",
        index.d
    )

    st.write(
        "Number of vectors after adding:",
        index.ntotal
    )


    # -----------------------------------
    # 8. Retrieve relevant chunks
    # -----------------------------------

    if question:

        question_embedding = model.encode(
            [question]
        )

        k = 3

        distances, indices = index.search(
            question_embedding.astype("float32"),
            k
        )


        st.write(
            "Distances:",
            distances.tolist()
        )

        st.write(
            "Indices:",
            indices.tolist()
        )


        # -----------------------------------
        # 9. Create context
        # -----------------------------------

        context = ""

        for i in indices[0]:

            context += chunks[i] + "\n\n"


        st.write("Retrieved Context:")

        st.write(context)


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

        response = requests.post(

            url,

            headers=headers,

            json=data,

            timeout=30
        )


        # -----------------------------------
        # 13. Check API response
        # -----------------------------------

        # response.raise_for_status()
        st.write("Status code:", response.status_code)
        st.write("API response:", response.text)


        # -----------------------------------
        # 14. Convert JSON response
        # -----------------------------------

        # result = response.json()


        # st.write("AI Response:")

        # st.write(result)

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        st.write("### 🤖 Answer")
        st.write(answer)
# 📚 AI Document Q&A

## 📌 Project Overview

AI Document Q&A is a Retrieval-Augmented Generation (RAG) application
that allows users to upload one or more PDF documents and ask questions
based only on their document content.

## ✨ Features

- Upload one or more PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate vector embeddings using Sentence Transformers
- Store and search embeddings using FAISS
- Retrieve relevant document content for a question
- Generate answers using an AI model
- Restrict answers to the retrieved document context
- Handle invalid or unreadable PDFs
- Handle AI API timeout and connection errors
- Protect API credentials using environment variables
- Interactive Streamlit interface


## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **PDF Upload**
   - The user uploads one or more PDF documents through the Streamlit interface.

2. **Text Extraction**
   - Text is extracted from each PDF using `PyPDF2`.

3. **Text Chunking**
   - The extracted text is divided into smaller chunks.
   - The application uses a chunk size of 500 characters with an overlap of 100 characters.
   - The overlap helps preserve context between neighboring chunks.

4. **Embedding Generation**
   - Each text chunk is converted into a numerical vector using the `all-MiniLM-L6-v2` Sentence Transformer model.
   - Each embedding contains 384 dimensions.

5. **FAISS Indexing**
   - The generated embeddings are stored in a FAISS index.
   - FAISS allows the application to efficiently search for chunks that are semantically similar to the user's question.

6. **Question Embedding**
   - When the user asks a question, the question is converted into an embedding using the same Sentence Transformer model.

7. **Semantic Retrieval**
   - FAISS compares the question embedding with the stored document embeddings.
   - The three most relevant chunks are retrieved.

8. **Relevance Check**
   - The application checks the distance of the most relevant result.
   - If the question does not appear to be sufficiently related to the uploaded documents, the application displays a warning instead of generating an answer.

9. **Context Construction**
   - The retrieved document chunks are combined into a context that is provided to the AI model.

10. **AI Answer Generation**
    - The context and user's question are sent to the Groq API.
    - The application uses the `openai/gpt-oss-120b` model.

11. **Document-Grounded Response**
    - The AI model is instructed to answer using only the retrieved context.
    - The generated answer is then displayed in the Streamlit interface.

### 🔄 RAG Pipeline

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Embeddings
      ↓
FAISS Vector Index
      ↓
User Question
      ↓
Question Embedding
      ↓
Semantic Search
      ↓
Relevant Chunks
      ↓
Context + Question
      ↓
Groq AI Model
      ↓
Document-Grounded Answer 
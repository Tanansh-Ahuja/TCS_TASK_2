# 🔍 RAG-based Q&A System using FastAPI + Groq + Chroma

This project implements a **Retrieval-Augmented Generation (RAG)** backend using:

- 🧠 **FastAPI** for the API
- 📚 **LangChain** for chaining logic
- 📦 **Chroma** for vector database storage
- 🤗 **HuggingFace** for embeddings
- ⚡ **Groq**'s blazing fast LLMs for answering questions

Upload any `.txt` or `.pdf` document and ask questions — the model retrieves the most relevant content and gives a contextual answer!

---

## ✅ Features

- 📄 Upload documents in `.txt` or `.pdf` format
- 🔗 Store and retrieve chunks using ChromaDB
- 🧠 Ask natural language questions
- 🤖 Powered by `llama3-70b-8192` model from Groq
- 💡 Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings

---

## 🧱 Architecture

```
Your Document (.pdf / .txt)
      ↓
LangChain Loaders & Text Splitters
      ↓
HuggingFace Embeddings (Vectorized)
      ↓
Chroma DB (for storage & search)
      ↓
User Question → Embedding → Similar Chunks
      ↓
LLM (Groq) with Context → Final Answer
```

---

## ⚠️ Issues Faced & Fixes

| Problem | Resolution |
|--------|------------|
| ❌ `mixtral-8x7b-32768` model decommissioned | ✅ Switched to `llama3-70b-8192` |
| ❌ `OpenAIEmbeddings` threw 401 error with Groq key | ✅ Used `HuggingFaceEmbeddings` instead |
| ❌ `GroqEmbeddings` not found | ✅ It's not a real class — use HuggingFace for embeddings |
| ❌ Only `.txt` supported initially | ✅ Added `.pdf` support using `PyMuPDFLoader` |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create a `.env` File

```env
GROQ_API_KEY=gsk_********************************
```

### 4️⃣ Run the Server

```bash
uvicorn main:app --reload
```

---

## 📤 Uploading a File

- **Endpoint:** `POST /upload/`
- **Body (form-data):**
  - `file`: Choose a `.pdf` or `.txt` file

✅ Example via **Postman** or any frontend.

---

## ❓ Asking a Question

- **Endpoint:** `POST /ask/`
- **Body (JSON):**
```json
{
  "question": "What is this document about?"
}
```

📩 Response:
```json
{
  "question": "What is this document about?",
  "answer": "The document discusses..."
}
```

---

## 📦 `requirements.txt`

```txt
fastapi
uvicorn
python-dotenv
langchain
langchain-community
langchain-core
langchainhub
langchain-groq
huggingface-hub
sentence-transformers
chromadb
PyMuPDF
```

---

## 🧪 Example Use Case

Upload a resume and ask:
> "What are the candidate's strengths?"

Or upload a project document and ask:
> "What are the key features mentioned?"

---

## 🧑‍💻 Author

Built by **Tanansh** for Task 2 of a Retrieval-Augmented Generation project using Groq's API.

---

## 📚 References

- [Groq Console](https://console.groq.com/)
- [LangChain Docs](https://docs.langchain.com/)
- [Chroma](https://docs.trychroma.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)


## 🏁 That's it!


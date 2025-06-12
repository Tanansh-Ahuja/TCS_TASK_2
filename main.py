from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from fastapi import Request
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain


load_dotenv()

app = FastAPI()

# Store vector DB in memory or on disk
CHROMA_PATH = "chroma_store"

def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext == ".pdf":
        print('got it, its a pdf')
        return PyMuPDFLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type. Only .txt and .pdf are supported.")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = f"uploaded_files/{file.filename}"
        print(f'filepath: {file_path}')
        os.makedirs("uploaded_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Step 1: Load the text
        documents = load_document(file_path)
        print(documents)

        # Step 2: Split text into chunks (to fit context limits)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        print("step 2 over")
        # Step 3: Embed and store in Chroma
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=CHROMA_PATH
        )
        vectorstore.persist()

        print("step 3 over")

        return {"message": "File ingested successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask/")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question")

        # Load Chroma DB with same embedding
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        # Step 1: Search similar docs
        docs = vectorstore.similarity_search(question)

        # Step 2: Ask LLM using the retrieved context
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )
        #chain = load_qa_chain(llm, chain_type="stuff")
        chain = load_qa_chain(llm,chain_type="map_reduce")
        try:
            answer = chain.run(input_documents=docs, question=question)
            return {"question": question, "answer": answer}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
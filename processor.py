import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
Eng_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
per_EMBEDDING_MODEL = "heydariAI/persian-embeddings"
VECTOR_STORE_PATH = "vector_store"

def extract_text_from_pdfs(pdf_docs):
    all_documents = []
    print("Extracting text from PDFs...")
    for pdf_doc in pdf_docs:
        with open(pdf_doc.name, "wb") as f:
            f.write(pdf_doc.getbuffer())
        
        loader = PyPDFLoader(pdf_doc.name)
        documents = loader.load()
        all_documents.extend(documents)
        
        os.remove(pdf_doc.name)
        
    print(f"Extracted {len(all_documents)} pages in total.")
    return all_documents

def chunk_text(documents):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_docs)} chunks.")
    return chunked_docs

def create_and_store_embeddings(chunks , language):
    print("Creating and storing embeddings...")
    if (language == "English"):
        embedding_model_name = Eng_EMBEDDING_MODEL
    elif (language == "Persian"):
        embedding_model_name = per_EMBEDDING_MODEL
    else:
        raise ValueError("Unsupported language.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("Embeddings stored successfully in ChromaDB.")
    return vector_store

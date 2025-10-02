# 📄 RAG Chatbot for PDF Documents

This project is a fully functional Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. It allows users to upload PDF documents, process them, and engage in a conversational chat where the AI's answers are grounded in the content of the uploaded files.



https://github.com/user-attachments/assets/d743482b-95e2-442f-afb1-72def77eb522



---
## ✨ Features

* Multi-Lingual Support: Choose between processing and chatting with documents in English or Persian.
* Multiple File Uploads: Upload one or more PDF documents at a time.
* Complete RAG Pipeline: Implements the full RAG workflow:
    1.  Load: Extracts text from PDF files.
    2.  Chunk: Splits text into smaller, semantically meaningful chunks.
    3.  Embed: Converts text chunks into vector embeddings using sentence-transformers.
    4.  Store: Saves embeddings in a local ChromaDB vector store.
    5.  Retrieve & Generate: Fetches relevant chunks based on a user's query and uses an LLM to generate a context-aware answer.
* Interactive UI: A simple, user-friendly web interface built with Streamlit.
* Modular Codebase: The project is organized into logical modules for UI (App.py), data processing (processor.py), and LLM logic (model.py).
* Flexible LLM Backend: Connects to LLMs via an API. The current implementation uses OpenRouter.ai to access models like Mistral.

---
## 🛠 Tech Stack & Dependencies

* Framework: LangChain, Streamlit
* LLM Service: OpenRouter.ai (via langchain-openai)
* Vector Database: ChromaDB
* Embeddings: sentence-transformers (via HuggingFaceEmbeddings)
  1. ٍEnglish embedding model: "all-MiniLM-L6-v2"
  2. Persian embedding model: "heydariAI/persian-embeddings"
* PDF Processing: pypdf
* Environment Management: python-dotenv

---
## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

* Python 3.11
* An API key from [OpenRouter.ai]

### 2. Installation & Setup

1.  Clone the repository (or set up your project folder):
    ```shell
    git clone [https://github.com/sabashafiei4/-Rag-PDF-Assistant.git]  
    cd -Rag-PDF-Assistant

    ```

2.  Create and activate a virtual environment:
    ```shell
       # Create the virtual environment
       python -m venv .venv

       # Activate on Windows
       .\.venv\Scripts\activate

       # Activate on macOS/Linux
        source .venv/bin/activate
    ```
    

3.  Install the required libraries:
    ```shell
    pip install -r requirements.txt
    ```
    

4.  Set up your API key:
    * Create a file in the root directory named .env.
    * Add your OpenRouter API key to this file:
        
        OPENROUTER_API_KEY="your_api_key_starts_with_sk-or-v1-..."
        

### 3. Running the Application

1.  Launch the Streamlit app from your terminal:
    ```shell
    streamlit run App.py
    ```
    
2.  Your web browser will open with the application running.

---
## 📖 Usage

1.  Select a Language: Choose either "English" or "Persian".
2.  Upload Documents: Upload one or more PDF files in the selected language.
3.  Process Documents: Click the "Process Documents" button. This will build the vector database.
4.  Chat: Once processing is complete, the chat interface will appear. Ask questions about the content of your documents.

## 📂 Project Structure

The project is organized into three main Python files for modularity:
```
├── App.py              # Main Streamlit UI and application flow
├── processor.py            # PDF processing, chunking, and vector store creation
├── model.py        # RAG chain setup, LLM, and prompt templates
└── requirements.txt    # Project dependencies

```


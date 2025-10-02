from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from the .env file

VECTOR_STORE_PATH = "vector_store"

def create_rag_chain(language: str):
    """
    Creates the RAG chain using the DeepSeek model from OpenRouter.
    """

    # 2. Set up models and prompts based on language
    if language == "English":
        embedding_model_name = "all-MiniLM-L6-v2"
        template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
    else: # Persian
        embedding_model_name = "heydariAI/persian-embeddings"
        template = """شما یک دستیار برای پاسخ به پرسش‌ها هستید. از قطعات متنی بازیابی شده زیر برای پاسخ به سوال استفاده کنید. اگر جواب را نمی‌دانید، فقط بگویید که نمی‌دانید. پاسخ را مختصر نگه دارید.
سوال: {question}
متن: {context}
پاسخ:"""

    # 3. Initialize the LLM using LangChain to connect to OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free", # Using the confirmed DeepSeek model
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )

    # 4. Load the vector store and create the retriever
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Build the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"✅ RAG chain created successfully for {language} using openAI.")
    return rag_chain
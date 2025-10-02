import streamlit as st
import shutil
from processor import extract_text_from_pdfs, chunk_text, create_and_store_embeddings
from model import create_rag_chain

VECTOR_STORE_PATH = "vector_store"

st.set_page_config(page_title="Chat with your PDFs 📄", layout="wide")
st.title("Chat with your PDFs 📄")

# --- Language Selection ---
st.header("Step 1: Choose a Language")
selected_language = st.radio(
    "Select the language of your documents:",
    ("English", "Persian"),
    index=None,
)

if selected_language:
    st.session_state.language = selected_language

# --- Main App Logic ---
if "language" in st.session_state:
    st.header(f"Step 2: Upload Your {st.session_state.language} Documents")

    
    uploaded_files = st.file_uploader(
        f"Upload your {st.session_state.language} PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )

    if "processed" not in st.session_state:
        st.session_state.processed = False

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
                
                language = st.session_state.language
                raw_documents = extract_text_from_pdfs(uploaded_files)
                chunked_documents = chunk_text(raw_documents)
                
                if not chunked_documents:
                    st.error("Could not extract any text from the uploaded PDF(s). Please ensure the file is not image-based or empty.")
                else:
                    create_and_store_embeddings(chunked_documents, language=language)
                    st.session_state.rag_chain = create_rag_chain(language=language)
                    
                    st.session_state.processed = True
                    st.session_state.messages = []
                    st.success(f"Processed {len(uploaded_files)} document(s) successfully!")

    if st.session_state.processed:
        st.header("Step 3: Chat with Your Documents")

        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please select a language to begin.")

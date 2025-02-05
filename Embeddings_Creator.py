import streamlit as st
import os
import zipfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Set Google API key as environment variable
os.environ["GOOGLE_API_KEY"] = google_api_key

# Sidebar configuration
with st.sidebar:
    st.title("PDF Embeddings Generator")
    st.markdown("Upload your PDFs to generate Embeddings and download them as a zip file.")

    # File uploader for multiple PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    # Process uploaded PDFs when the button is clicked
    if uploaded_files:
        with st.spinner("Processing Documents... Please wait."):
            def vector_embedding(uploaded_files):
                if "vectors" not in st.session_state:
                    # Initialize Embeddings if not already done
                    st.session_state.Embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001"
                    )
                    all_docs = []

                    # Process each uploaded file
                    for uploaded_file in uploaded_files:
                        # Save the uploaded file temporarily
                        with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                        ) as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name

                        # Load the PDF document
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()  # Load document content

                        # Remove the temporary file
                        os.remove(temp_file_path)

                        # Add loaded documents to the list
                        all_docs.extend(docs)

                    # Split documents into manageable chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    final_documents = text_splitter.split_documents(all_docs)

                    # Create a vector store with FAISS
                    st.session_state.vectors = FAISS.from_documents(
                        final_documents, st.session_state.Embeddings
                    )

                    # Save the FAISS index to the "Embeddings" folder
                    if not os.path.exists("Embeddings"):
                        os.makedirs("Embeddings")
                    st.session_state.vectors.save_local("Embeddings")
                    st.sidebar.success("Embeddings Successfully Created! 🎉")

                    # Create a zip file of the Embeddings
                    with zipfile.ZipFile('Embeddings.zip', 'w') as zipf:
                        for root, dirs, files in os.walk("Embeddings"):
                            for file in files:
                                zipf.write(os.path.join(root, file),
                                           os.path.relpath(os.path.join(root, file),
                                                           os.path.join("Embeddings", '..')))

                    st.session_state.Embeddings_ready = True

            vector_embedding(uploaded_files)

    # Download button for Embeddings zip file
    if st.session_state.get("Embeddings_ready", False):
        with open("Embeddings.zip", "rb") as fp:
            btn = st.download_button(
                label="Download Embeddings",
                data=fp,
                file_name="Embeddings.zip",
                mime="application/zip"
            )
        if btn:
            st.sidebar.success("Embeddings downloaded successfully!")

# Main content area
st.title("PDF Embeddings Generator")
st.markdown("""
    Welcome to the PDF Embeddings Generator! 
    Use the sidebar to upload your PDFs and generate Embeddings. 
    Once the Embeddings are created, you can download them as a zip file.
""")

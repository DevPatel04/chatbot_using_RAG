from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import streamlit as st
import tempfile
import logging
import chromadb

logging.basicConfig(level=logging.INFO)
import sqlite3
print("SQLite version:", sqlite3.sqlite_version)




def setup_chroma_db(docs, embeddings):
    
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    vectorstore = Chroma.from_documents(
        collection_name="dev",
        documents=docs,
        embedding=embeddings,
        persist_directory="."
    )

    return vectorstore



def main():
    st.title("RAG File Reader")
    st.write("Upload a PDF file and ask questions about its content.")

    model_name = st.selectbox("Select the model", ["llama-3.1-8b-instant","gemma2-9b-it","mixtral-8x7b-32768"] )

    if model_name is not None:
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'docx', 'doc'])


        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_file_path = tmp_file.name

            # Load and split document into chunks
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(temp_file_path)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                loader = Docx2txtLoader(temp_file_path)
            else:
                st.error("Unsupported file type.")
                return
            pages = loader.load()

            r_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = r_splitter.split_documents(pages)

            if not docs:
                st.error("No content extracted from the PDF. Please upload a valid document.")
                return

            # Initialize Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=st.secrets['GOOGLE_API_KEY']
            )

            # Create Chroma Vector Store
            vectorstore = setup_chroma_db(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

            # Initialize LLM
            llm = ChatGroq(
                model=model_name,
                temperature=0.1,
            )

            # Conversation Memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # Retrieval-Augmented Generation Chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                verbose=True
            )

            query = st.text_input("Ask a question about the document:")

            if st.button("Submit"):
                if query:
                    response = qa_chain.run(query)
                    st.write("Answer:", response)
                else:
                    st.write("Please enter a question.")


if __name__ == "__main__":
    main()

import streamlit as st
import os
import tempfile
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit session setup
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="RAG Chatbot", page_icon="🧠")
st.title("🧠 Gemini RAG Chatbot")
st.markdown("Upload PDFs, CSVs, or TXTs and chat with them using **Gemini 2.0 Flash**.")

uploaded_files = st.file_uploader(
    "📎 Upload your documents",
    type=["pdf", "csv", "txt"],
    accept_multiple_files=True
)

def load_documents(files):
    documents = []
    for file in files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif file.name.endswith(".csv"):
            loader = CSVLoader(temp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            continue

        documents.extend(loader.load())
    return documents

if st.session_state.chain is None and st.button("🔄 Process Documents") and st.session_state.get("uploaded_files"):
    with st.spinner("Processing..."):
        docs = load_documents(st.session_state.uploaded_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # Gemini Flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        st.session_state.chain = chain
        st.success("✅ Documents indexed. You can start chatting!")

# Store uploaded files in session
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.chain:
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        with st.spinner("Thinking..."):
            result = st.session_state.chain.invoke({"question": user_input})

            # Add to chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", result["answer"]))

            # Debug print
            print("\n🧠 Answer:", result["answer"])
            print("\n📚 Source Documents:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n--- Source {i} ---")
                print(doc.page_content[:500])
                print(f"Metadata: {doc.metadata}")

            for role, msg in st.session_state.chat_history:
                with st.chat_message(role):
                    st.markdown(msg)

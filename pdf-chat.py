# pdf-chat.py
#  python -m streamlit run pdf-chat.py

import os
import streamlit as st
from dotenv import load_dotenv

# ———— LangChain & related imports ————
from langchain import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# (Optional) type hint for history
from langchain.schema import BaseChatMessageHistory
# ———— END imports ————

# Load environment variables (e.g., your HuggingFace token)
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ———— Set up embeddings + Streamlit UI ————
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")

st.title("Conversational RAG with PDF Uploads + History")
st.write("Upload a PDF below and then ask questions about its contents.")

# Step 1: ask for Groq API key and instantiate the LLM
api_key = st.text_input("Enter your Groq API key:", type="password")
if not api_key:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

# ———— Keep chat histories in session_state ————
if "history_store" not in st.session_state:
    st.session_state.history_store = {}  # maps session_id → ChatMessageHistory

def get_history(session_id: str) -> ChatMessageHistory:
    """Return a ChatMessageHistory for this session, creating it if it doesn’t exist."""
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

# ———— Upload & Index PDF(s) ————
uploaded_file = st.file_uploader(
    "Choose a single PDF file:", type="pdf", accept_multiple_files=False
)

vectorstore = None
base_retriever = None

if uploaded_file:
    # 1) Save the PDF locally
    temp_path = "./temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # 2) Load into LangChain Documents
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # 3) Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 4) Embed & build Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    base_retriever = vectorstore.as_retriever()

    st.success("✅ PDF processed and indexed successfully!")

# ———— Prepare prompts & chains if PDF is indexed ————
condense_chain = None
answer_chain = None

if base_retriever:
    # A) Prompt template to condense “history + new question” → standalone question
    condense_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a question‐rephraser. Given the chat history and the user's new question "
             "(which might refer to earlier messages), formulate a standalone question that can be "
             "understood without prior context. Do NOT answer—just rephrase if needed; otherwise "
             "return the question unchanged."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    condense_chain = LLMChain(llm=llm, prompt=condense_prompt)

    # B) Prompt template to take (retrieved docs + chat history + question) → concise answer
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are an assistant for question‐answering tasks. Use the following retrieved context "
             "to answer the question. If you don't know, say so. Keep the answer to at most three "
             "sentences and remain concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    # Build the “stuff documents” chain (returns a RunnableSequence)
    answer_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)

# ———— Session ID & User Q&A ————
session_id = st.text_input("Session ID", value="default_session")
history = get_history(session_id)

if base_retriever and condense_chain and answer_chain:
    user_question = st.text_input("Your question:", placeholder="Ask something about the PDF…")
    if user_question:
        # 1) Append user question into history
        history.add_user_message(user_question)

        # 2) If there is prior chat history, run condense_chain to get a standalone query
        if len(history.messages) > 0:
            condense_result = condense_chain.invoke({
                "chat_history": history.messages,
                "input": user_question
            })
            # Extract text if returned as dict, else assume string
            if isinstance(condense_result, dict):
                standalone_query = condense_result.get("text", "").strip()
            else:
                standalone_query = condense_result.strip()
        else:
            standalone_query = user_question

        # 3) Retrieve relevant document chunks using the standalone query
        docs = base_retriever.get_relevant_documents(standalone_query)

        # 4) Run the “stuff documents” chain to get a concise answer
        answer_result = answer_chain.invoke({
            "input": standalone_query,
            "context": docs,
            "chat_history": history.messages
        })
        # Extract text if returned as dict, else convert to string
        if isinstance(answer_result, dict):
            answer = answer_result.get("text", "").strip()
        else:
            answer = str(answer_result).strip()

        # 5) Append the assistant’s answer into history
        history.add_ai_message(answer)

        # 6) Display the assistant’s answer and the last 10 chat messages
        st.write("**Assistant:**", answer)
        st.write("---")
        st.write("**Chat History (latest 10 messages):**")
        for msg in history.messages[-10:]:
            role = msg.type       # “human” or “ai”
            text = msg.content    # use .content instead of msg.data["content"]
            st.write(f"- _{role}_: {text}")
else:
    if not uploaded_file:
        st.info("Upload a PDF to get started.")

import streamlit as st
from langchain_groq import ChatGroq
import faiss
import numpy as np
from PyPDF2 import PdfReader
import tempfile
from fastembed import TextEmbedding

# --- UI Setup ---
st.set_page_config(page_title=" PDF Chatbot", page_icon="📄", layout="centered")
st.markdown("<h1 style='text-align: center;'>📄 Chat with PDF</h1>", unsafe_allow_html=True)

# --- API Key ---
api_key = "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"

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_model" not in st.session_state and api_key:
    st.session_state.chat_model = ChatGroq(model="gemma2-9b-it", api_key=api_key)

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF to chat with it", type=["pdf"])

# --- Extract Text from the First Page of PDF ---
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    document_text = ""
    if len(pdf_reader.pages) > 0:  # Check if the PDF has at least one page
        document_text = pdf_reader.pages[0].extract_text()  # Extract text from the first page
    return document_text

# --- Process PDF and Generate Embeddings ---
def process_pdf(pdf_path):
    # Extract text from the first page of the PDF
    document_text = extract_text_from_pdf(pdf_path)

    # Split the text into smaller chunks for embedding
    chunk_size = 500  # Adjust chunk size as needed
    chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]

    # Generate embeddings for the chunks
    embed_model = TextEmbedding()  # Initialize the embedding model
    embeddings = list(embed_model.embed(chunks))  # Convert generator to list

    # Create a FAISS index for the embeddings
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return chunks, index, embed_model

# --- RAG Function ---
def rag_ask(query, chunks, index, embed_model, top_k=2):
    query_embedding = list(embed_model.embed([query]))[0]  # Convert generator to list and get the first embedding
    D, I = index.search(np.array([query_embedding]), top_k)
    retrieved = [chunks[i] for i in I[0]]

    context = "\n".join(retrieved)
    prompt = f"""Use the following context to answer the question:
Context:
{context}

Question: {query}
Answer:"""
    return st.session_state.chat_model.invoke(prompt)

pdf = False  # Flag to check if PDF is processed

# --- Process PDF if uploaded ---
if uploaded_file and "retriever" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Process the PDF to extract text, generate embeddings, and create an index
    chunks, index, embed_model = process_pdf(pdf_path)
    # Store the chunks, index, and embedding model in session state
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.embed_model = embed_model

    # Notify user that PDF is ready
    st.info("✅ PDF processed! Now you can ask your questions about the document.")

# --- Input Box for User ---
user_input = st.chat_input("Ask questions about the PDF or chat normally...")

if user_input:
    # Handle user input for normal chat or PDF-related chat
    if uploaded_file and "chunks" in st.session_state:
        response = rag_ask(user_input, st.session_state.chunks, st.session_state.index, st.session_state.embed_model)
    else:
        response = st.session_state.chat_model.invoke(user_input)
    
    # Always extract the content if available, otherwise use the response itself
    response_text = getattr(response, "content", response)

    # Display user and assistant messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response_text)  # Display only the content

    # Store the conversation history
    st.session_state.chat_history.append({"user": user_input, "bot": response_text})

# --- Display chat history (user and assistant messages) ---
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

# --- API Key Check ---
if not api_key:
    st.warning("Please enter your ChatGroq API Key to proceed.")
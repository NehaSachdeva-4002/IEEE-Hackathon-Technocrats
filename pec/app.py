import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# --- PDF processing ---
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_text_chunks(text):
    """Splits large text into smaller chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_local_embeddings(text_chunks):
    """Generates embeddings and creates FAISS vectorstore."""
    # Initialize HuggingFace embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings for each text chunk
    embeddings = [embeddings_model.embed_query(chunk) for chunk in text_chunks]

    # Create FAISS vector store from embeddings and metadata (texts)
    vectorstore = FAISS.from_texts(text_chunks, embeddings_model)
    return vectorstore

# --- Load QA model locally ---
def load_qa_pipeline():
    """Loads a local HuggingFace QA model."""
    model = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model)
    return pipeline("question-answering", model=qa_model, tokenizer=tokenizer)

# --- Handle user input ---
def handle_userinput(user_question, qa_pipeline, vectorstore):
    """Finds the most relevant chunk and answers the user's question."""
    # Perform similarity search on the vectorstore
    docs = vectorstore.similarity_search(user_question, k=1)

    # Use the QA model to answer based on the retrieved document
    result = qa_pipeline(question=user_question, context=docs[0].page_content)
    return result["answer"]

# --- Main Application ---
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    # Initialize vectorstore in session state if not already present
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Handle user input if PDFs are processed and question is asked
    if user_question and st.session_state.vectorstore:
        qa_pipeline = load_qa_pipeline()
        answer = handle_userinput(user_question, qa_pipeline, st.session_state.vectorstore)

        # Use st.markdown for better formatting of the answer
        st.markdown(f"*Answer:*<br>{answer.replace('\n', '<br>')}", unsafe_allow_html=False)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                # Extract text and create embeddings
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_local_embeddings(text_chunks)

                # Display the parsed document text
                st.write("Parsed Document Text:")
                st.write(raw_text)  # Display the entire extracted text

if __name__ == '__main__':
    main()
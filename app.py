import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="📚 PDF Q&A Assistant", layout="wide")
st.title("📚 Course Material Q&A Assistant")
st.markdown("Upload your PDFs and ask questions powered by **Falcon-7B-Instruct**!")

# ==============================
# Sidebar: Settings
# ==============================
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("🔄 Reset FAISS Index"):
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        st.sidebar.success("FAISS index reset successfully!")
    else:
        st.sidebar.info("No FAISS index found to reset.")

model_choice = st.sidebar.selectbox(
    "Choose LLM:",
    ["Falcon-7B-Instruct", "GPT-Neo-125M"],
    index=0
)

# ==============================
# PDF Upload & Text Extraction
# ==============================
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

all_texts = []
if uploaded_files:
    progress = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            pdf = PdfReader(file_path)
            pdf_text = ""
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                except Exception:
                    text = None
                if text:
                    pdf_text += f"[Page {page_num}]\n{text}\n"
            if pdf_text.strip():
                all_texts.append(pdf_text)
            else:
                st.warning(f"⚠️ {uploaded_file.name} had no extractable text.")
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
        progress.progress((i + 1) / len(uploaded_files))
    if all_texts:
        st.success(f"✅ {len(all_texts)} PDF(s) processed successfully!")
    else:
        st.info("No usable text extracted.")

# ==============================
# Split Text into Chunks
# ==============================
docs = []
if all_texts:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for text in all_texts:
        docs.extend(text_splitter.split_text(text))
    st.info(f"Text split into {len(docs)} chunks.")

# ==============================
# FAISS Vector Store
# ==============================
VECTORSTORE_PATH = "faiss_index"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = None
if docs:
    try:
        if os.path.exists(VECTORSTORE_PATH):
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
            vectorstore.add_texts(docs)
            vectorstore.save_local(VECTORSTORE_PATH)
        else:
            vectorstore = FAISS.from_texts(docs, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
        st.success("✅ FAISS vector store ready!")
    except Exception as e:
        st.error(f"Error building/loading FAISS index: {e}")
elif os.path.exists(VECTORSTORE_PATH):
    try:
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
        st.success("✅ Loaded existing FAISS vector store.")
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")

# ==============================
# LLM Setup
# ==============================
qa = None
if vectorstore is not None:
    try:
        if model_choice == "Falcon-7B-Instruct":
            # Use the valid Falcon-7B-Instruct model
            llm_pipeline = pipeline(
                "text-generation",
                model="tiiuae/falcon-7b-instruct",
                trust_remote_code=True,
                max_new_tokens=256,
                temperature=0.7,
                device=-1  # CPU only; change to 0 for GPU
            )
        else:
            llm_pipeline = pipeline(
                "text-generation",
                model="EleutherAI/gpt-neo-125M",
                max_new_tokens=256,
                temperature=0.7,
                device=-1
            )

        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
else:
    st.info("No FAISS index available. Upload PDFs to create one.")

# ==============================
# Q&A Interface
# ==============================
st.markdown("### ❓ Ask Questions")

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

question = st.text_input("Enter your question:")

if question:
    if qa is None:
        st.error("Q&A pipeline not ready. Upload PDFs first.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = qa.run(question)
            except Exception as e:
                answer = f"Error while generating answer: {e}"
        st.markdown(f"### **Answer:** {answer}")
        st.session_state.qa_history.append({"question": question, "answer": answer})

if st.session_state.qa_history:
    st.markdown("### 📝 Q&A History")
    for i, item in enumerate(reversed(st.session_state.qa_history[-10:]), 1):
        st.write(f"**Q{i}.** {item['question']}")
        st.write(f"**A{i}.** {item['answer']}")
        st.write("---")

# Course Material Q&A Assistant Project

<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/streamlit.svg" alt="Logo" width="90" height="90">
</p>

<h1 align="center">📚 PDF Q&A Assistant</h1>

<p align="center">
  <i>Ask intelligent, context-aware questions from your PDFs using Retrieval-Augmented Generation (RAG)</i>  
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-red.svg?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-✨-purple.svg?style=for-the-badge&logo=chainlink" alt="LangChain">
  <img src="https://img.shields.io/badge/HuggingFace-🤗-yellow.svg?style=for-the-badge&logo=huggingface" alt="Hugging Face">
  <img src="https://img.shields.io/badge/FAISS-Facebook-blue.svg?style=for-the-badge&logo=facebook" alt="FAISS">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-screenshots">Screenshots</a> •
  <a href="#-license">License</a>
</p>

---

## 🚀 Why This Project?

Large PDF documents (textbooks, research papers, manuals) are **hard to navigate**. Traditional keyword search (`CTRL+F`) fails when queries are phrased differently.  

This app solves that by combining **semantic vector search** with **LLMs** → giving you **contextual, accurate answers** instead of keyword matches.  

✨ **Core Idea:**  
- Use **RAG (Retrieval-Augmented Generation)** → retrieval + LLM reasoning  
- Store embeddings in **FAISS** for semantic search  
- Generate answers with **Falcon-7B-Instruct** or **GPT-Neo-125M**  
- Smooth **Streamlit UI** for an end-user friendly interface  

---

## ✨ Features

- 📂 Upload **multiple PDFs** at once  
- 🔎 Automatic **text chunking + embeddings**  
- 💾 Persistent **FAISS index** (reset anytime)  
- 🧠 **LLM-powered Q&A** (Falcon-7B or GPT-Neo-125M)  
- 📝 **Q&A history** maintained across a session  
- 🎛️ Model switcher:  
  - **Falcon-7B-Instruct** → powerful, accurate answers  
  - **GPT-Neo-125M** → lightweight fallback for free-tier CPU  

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LangChain](https://www.langchain.com/) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **Embeddings** | [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| **LLMs** | [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct), [GPT-Neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) |
| **Deep Learning** | [PyTorch](https://pytorch.org/) |
| **PDF Parsing** | [PyPDF2](https://pypi.org/project/PyPDF2/) |

---

## 📂 Project Structure

```

.
├── app.py              # Main Streamlit application
├── requirements.txt    # High-level dependencies
├── constraints.txt     # Pinned versions for reproducibility
├── README.md           # Documentation
└── uploaded_pdfs/      # User-uploaded PDFs

````

---

## 🔧 Installation

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/apoorvad444/GenAI.git
cd pdf-qa-assistant

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt -c constraints.txt
````

---

## ▶️ Usage

Run the app:

```bash
streamlit run app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🚀 Deployment

### 🔴 Streamlit Cloud

1. Push repo to GitHub
2. Deploy on **Streamlit Cloud**
3. ⚠️ Use **GPT-Neo-125M** for free CPU tier (Falcon-7B may exceed memory)

### 🤗 Hugging Face Spaces

1. Create a new **Streamlit Space**
2. Select a **GPU runtime** (T4/A10/A100 recommended)
3. Push this repo → Falcon-7B works smoothly

---


### 📑 Example Workflow

1. Upload your course material PDFs
2. App extracts & chunks text
3. FAISS builds a vector index
4. Ask questions → relevant chunks retrieved → LLM generates contextual answers

![Workflow Screenshot](https://via.placeholder.com/800x400?text=Workflow)



## 📜 License

Distributed under the **MIT License**.
See [LICENSE](./LICENSE) for details.



## 👨‍💻 Author

Developed with ❤️ by **[Abhinav Jajoo,
Apporva Dixit,Vrinda Sharma]**


## 🙌 Acknowledgements

* [Falcon LLM](https://huggingface.co/tiiuae/falcon-7b-instruct)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [LangChain](https://www.langchain.com/)
* [Streamlit](https://streamlit.io/)
* [SBERT](https://www.sbert.net/)

```


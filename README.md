# 🎯 Marketing & Enterprise Hybrid RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pinecone](https://img.shields.io/badge/vector_db-Pinecone_Serverless-purple.svg)](https://www.pinecone.io/)
[![Ollama](https://img.shields.io/badge/local_llm-Ollama_(Llama_3.2)-black.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interview-grade, production-grounded **Hybrid Retrieval-Augmented Generation (RAG)** pipeline designed to eliminate retrieval failure modes on real-world multi-page corporate and marketing documents.

Unlike standard tutorial RAG that relies exclusively on naive vector search, this system combines **Dense Semantic Search (Pinecone)** and **Sparse Lexical Search (BM25)** via **Reciprocal Rank Fusion (RRF)**, re-ranks candidate passages using a **Cross-Encoder (FlashRank)**, and includes an empirical **Evaluation Suite** to measure Precision@K and Faithfulness.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    D[📄 Multi-Page PDF Documents] --> L[PyPDFLoader<br/>Extract Text & Page Metadata]
    L --> C[Recursive Character Text Splitter<br/>Chunk Size: 600 | Overlap: 100]
    
    C --> V[(Pinecone Cloud<br/>Dense Embeddings)]
    C --> B[(BM25 Sparse Index<br/>Exact Lexical Frequencies)]
    
    Q[🔍 User Query] --> V
    Q --> B
    
    V -- Top-8 Semantic Chunks --> RRF[Reciprocal Rank Fusion<br/>Score = 1 / (60 + rank)]
    B -- Top-8 Keyword Chunks --> RRF
    
    RRF --> CE[FlashRank Cross-Encoder<br/>Joint Query-Doc Scoring]
    CE --> T3[Top-3 High-Signal Chunks]
    
    T3 --> P[Grounded Prompt Template<br/>Zero-Hallucination Guardrails]
    P --> O[Local Ollama LLM<br/>llama3.2:3b]
    O --> A[Grounded Answer + Page Citations]
    
    A --> E1[Precision@K Evaluation]
    A --> E2[Faithfulness Evaluation]
```

---

## 💡 Why Hybrid RAG? (The Core Problem)

Standard vector search (`similarity_search()`) fails on two critical enterprise tasks:
1. **Exact Numbers & Financial Codes:** Dense embeddings compress an entire 500-token chunk into a single vector, blurring exact figures (e.g., `$119.58 billion` vs `$117.15 billion`) or specific fiscal quarters (`Q1 2024` vs `Q1 2023`).
2. **Specific Keywords & Brand Tags:** Rare product names, compliance rules, and alphanumeric codes often have weak representations in generic embedding models.

### How this system solves it:
* **BM25 (Sparse):** Accurately catches exact keywords, quarters, and accounting metrics using term frequency / inverse document frequency.
* **Dense Embeddings:** Captures semantic meaning, synonyms, and high-level conceptual questions.
* **Reciprocal Rank Fusion (RRF):** Merges uncalibrated scores from both retrieval systems purely by rank position ($k=60$).
* **Cross-Encoder Reranker:** Evaluates token-level cross-attention between the query and candidate passages to ensure the top-3 chunks fed to the LLM have the highest information density.

---

## 📁 Repository Structure

```text
marketing-hybrid-rag/
│
├── data/
│   ├── apple_fy24_q1.pdf        # Official Apple FY24 Q1 Financial Statements
│   ├── apple_fy23_q4.pdf        # Official Apple FY23 Q4 Operations & Annual Filings
│   └── apple_fy23_q3.pdf        # Official Apple FY23 Q3 Financial Operations
│
├── hybrid_rag.ipynb             # Master implementation & evaluation notebook
├── requirements.txt             # Minimal, lightweight dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Ignore .env, caches, and checkpoints
└── README.md                    # Project documentation
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language & Environment** | Python 3.10+, Jupyter Notebook | Interactive, step-by-step experimentation |
| **Document Processing** | `pypdf`, `langchain-community` | Text extraction and page metadata preservation |
| **Dense Vector Search** | Pinecone Serverless | Fast, scalable semantic similarity search |
| **Sparse Lexical Search** | `rank-bm25` | Exact keyword and alphanumeric matching |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) | Rank-based score calibration |
| **Cross-Encoder Reranker**| FlashRank (`ms-marco-MiniLM-L-12-v2`) | Joint query-passage cross-attention |
| **Local Reasoning LLM** | Ollama (`llama3.2:3b`) | Privacy-preserving, local answer generation |
| **Evaluation Metrics** | Precision@K, Faithfulness | Quantitative retrieval quality measurement |

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Harshsinghrajpurohit/marketing-hybrid-rag.git
cd marketing-hybrid-rag
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=apple-hybrid-rag
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
EMBEDDING_MODEL=nomic-embed-text:latest
```

### 4. Run the Notebook
Make sure Ollama is running locally:
```bash
ollama run llama3.2:3b
```

Launch Jupyter:
```bash
jupyter notebook hybrid_rag.ipynb
```

---

## 📊 Quantitative Evaluation

The notebook contains an empirical evaluation section testing a curated benchmark of questions:
* **Precision@3 Benchmark:** Comparing Retrieval Precision across **Dense-only**, **BM25-only**, **Hybrid (RRF)**, and **Hybrid + Reranking**.
* **Faithfulness Metric:** Evaluating whether generated answers are strictly derived from the retrieved source context without hallucinations.
* **Source Citations:** Every answer returns the exact document name and page number.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

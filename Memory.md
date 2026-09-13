# Project Memory & Learning Log (Memory.md)

*This file tracks decisions, technical concepts, problems solved, and learning milestones for the Hybrid RAG System with Reranking & Evaluation. It is updated incrementally throughout the project.*

---

## 1. Project Overview & Objective
* **Project Name:** Hybrid RAG System with Reranking & Evaluation
* **Target Role:** AI / LLM Engineer (Fresher / Intern / Entry-Level)
* **Core Philosophy:** Minimal, observable, defensible code inside a single Jupyter Notebook (`hybrid_rag.ipynb`). Avoid overengineering (no microservices, no multi-agents, no separate ML models).
* **Dataset/Corpus:** 3 official public multi-quarter financial and operational reports from Apple Inc. (`data/apple_fy24_q1.pdf`, `data/apple_fy23_q4.pdf`, `data/apple_fy23_q3.pdf`). Provides both narrative business performance text (for semantic dense search) and exact accounting metrics/quarters (for BM25 lexical search).

---

## 2. Core Architecture Decisions
1. **Document Loading:** `PyPDFLoader` to extract raw text and preserve critical metadata (`source`, `page`).
2. **Chunking Strategy:** `RecursiveCharacterTextSplitter` with explicit chunk size and overlap to preserve table context across page boundaries.
3. **Dense Semantic Retrieval:** Local Ollama embeddings (`nomic-embed-text:latest`, 768-dim) indexed into Pinecone Serverless. Excels at semantic meaning, intent, and synonyms.
4. **Sparse Lexical Retrieval:** `rank-bm25` built over the exact same chunks. Excels at exact keywords, fiscal quarters ("Q1 2024"), and line-item names.
5. **Fusion Layer:** Reciprocal Rank Fusion (RRF with $k=60$) to merge dense and sparse candidate rankings without score normalization issues.
6. **Reranker:** FlashRank Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) to joint-score `(query, passage)` pairs and select the top 3 highest-signal chunks.
7. **Generation & Grounding:** Local Ollama LLM (`llama3.2:3b`) with a strict context-bounded prompt returning verified source citations (`document.pdf — page X`).
8. **Evaluation:** Quantitative benchmark testing **Precision@K** across Dense, BM25, Hybrid, and Hybrid + Reranker, plus a simple **Faithfulness** check.

---

## 3. Active Phase Tracker

| Step / Section | Description | Status | Key Milestones & Notes |
| :--- | :--- | :--- | :--- |
| **Step 1** | Architecture & Rules Alignment | ✅ Completed | Agreed on human-in-the-loop pair programming, Jupyter notebook format, and core pillars. |
| **Step 2** | Environment & Data Setup | ✅ Completed | Downloaded 3 official Apple reports to `data/`, configured `requirements.txt`, `.env`, `.gitignore`. |
| **Section 1 & 2** | Notebook Overview & Env Verification | ⏳ Next | Initialize `hybrid_rag.ipynb`, verify package imports, Ollama, and Pinecone connectivity. |
| **Section 3** | Load Documents & Inspect Metadata | ⏳ Pending | Load PDFs, inspect `Document` structure (`page_content`, `metadata`). |
| **Section 4** | Text Chunking & Inspection | ⏳ Pending | Split documents, test chunk sizes, inspect boundary handling. |
| **Section 5** | Dense Retrieval (Pinecone) | ⏳ Pending | Embed with Nomic, upsert to Pinecone, run semantic search queries. |
| **Section 6** | BM25 Lexical Retrieval | ⏳ Pending | Build BM25 index, test keyword queries against dense results. |
| **Section 7** | Reciprocal Rank Fusion (RRF) | ⏳ Pending | Implement RRF formula, compare fused ranks against individual ranks. |
| **Section 8** | Cross-Encoder Reranking | ⏳ Pending | Add FlashRank, measure ranking order shift. |
| **Section 9** | RAG Generation with Citations | ⏳ Pending | Prompt engineering, zero-hallucination guardrail, output citations. |
| **Section 10 & 11**| Evaluation (Precision@K & Faithfulness) | ⏳ Pending | Curate test questions, calculate real metrics without fabricating data. |
| **Section 12** | Failure Analysis & Interview Prep | ⏳ Pending | Analyze where retrieval failed and why; document interview Q&As. |

---

## 4. Key Learnings & Interview Talking Points (Living Notes)
* *Why not rely solely on Dense Vector Search?* Dense embeddings compress 500 words into 768 floating-point numbers. In doing so, exact numbers (like "$119.58 billion") and specific codes get smoothed out. BM25 directly counts exact term frequency, making it essential for financial and enterprise document search.
* *Why RRF instead of linear score combination?* Cosine similarity is bounded [0, 1] while BM25 is unbounded [0, $\infty$]. Adding them directly creates a biased, uncalibrated score. RRF relies exclusively on relative rank positions.
## 5. Documentation & File Strategy
* **Decision (confirmed 13-09-2026):** `Memory.md` is the **only** living/maintained document for this project.
* The 5 spec documents suggested by `rules for a project.txt` (PRD.md, Architecture.md, Rules.md, Phases.md, Design.md) are **intentionally NOT created** — architecture, phases, and allowed-libraries live inside this Memory.md instead, keeping the repo minimal per the core philosophy ("minimal, observable, defensible code").
* Any durable decision made during pair programming gets recorded here so context survives chat/tool switches.

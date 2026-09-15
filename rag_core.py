# rag_core.py — hybrid RAG engine shared by the FastAPI (api.py) and Streamlit (ui.py) apps.
# Extracted from hybrid_rag.ipynb (Sections 3-9). State built ONCE per process.

import os
from dotenv import load_dotenv
load_dotenv()

DATA_FILES = [
    "data/apple_fy24_q1.pdf",
    "data/apple_fy23_q4.pdf",
    "data/apple_fy23_q3.pdf",
]
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "apple-hybrid-rag")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_state = None  # module-level cache: built once, shared by every request


def get_state():
    global _state
    if _state is None:
        _state = _build_state()
    return _state


def _build_state():
    # Heavy imports live HERE (lazy) so `import rag_core` / API startup stays fast (~2s).
    # They only load once, on the FIRST /query, together with the expensive state build.
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from pinecone import Pinecone
    from flashrank import Ranker, RerankRequest

    # 1) Load + clean + chunk (notebook Sections 3-4, incl. the $ glyph fix)
    docs = []
    for path in DATA_FILES:
        for page in PyPDFLoader(path).load():
            page.page_content = page.page_content.replace("/dollarsign", "$")
            docs.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Header propagation — the retrieval fix that took P@3 from 0.00 to 0.28
    first = {}
    for cid, c in enumerate(chunks):
        key = (c.metadata["source"], c.metadata["page"])
        if key in first and cid != first[key]:
            c.page_content = chunks[first[key]].page_content + "\n" + c.page_content
        else:
            first[key] = cid

    # 2) Dense arm         3) Sparse arm
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(PINECONE_INDEX)
    bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])

    # 4) Reranker + LLM
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, base_url=OLLAMA_URL)

    return {"chunks": chunks, "embeddings": embeddings, "index": index,
            "bm25": bm25, "ranker": ranker, "llm": llm}


def _dense(state, query, k=3):
    ms = state["index"].query(vector=state["embeddings"].embed_query(query),
                              top_k=k, include_metadata=True).matches
    return [(int(m.id.split("-")[1]), m.score) for m in ms]


def _bm25(state, query, k=3):
    scores = state["bm25"].get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in ranked]


def _rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _merge(state, query, k=3):
    return _rrf([[i for i, _ in _dense(state, query, k)],
                 [i for i, _ in _bm25(state, query, k)]])[:k]


def _rerank(state, query, candidates, top_n=3):
    from flashrank import RerankRequest          # ← add this line
    passages = [{"text": state["chunks"][i].page_content, "meta": i} for i in candidates]
    results = state["ranker"].rerank(RerankRequest(query=query, passages=passages))
    return [(int(r["meta"]), float(r["score"]), j + 1) for j, r in enumerate(results[:top_n])]


def answer(question, top_n=3):
    state = get_state()
    ranked = _rerank(state, question, [i for i, _ in _merge(state, question, 5)], top_n)

    blocks, sources = [], []
    for cid, _, _ in ranked:
        blocks.append(f"[{len(sources) + 1}] {state['chunks'][cid].page_content}")
        sources.append(f"{state['chunks'][cid].metadata['source']} — page {state['chunks'][cid].metadata['page']}")

    prompt = f"""You are a financial analyst assistant. Answer ONLY from the context below.
Do not reproduce the provided context text.

CONTEXT:
{chr(10).join(blocks)}

QUESTION: {question}

Rules:
- If the context does not contain the answer, say "I could not find sufficient information".
- Cite each fact as [N] matching its context block.

ANSWER:"""

    response = state["llm"].invoke(prompt)
    return {"question": question, "answer": response.content, "sources": sources}


if __name__ == "__main__":
    result = answer("What were Apple's total net sales for the three months ended December 30, 2023?")
    print("ANSWER:", result["answer"])
    print("SOURCES:", result["sources"])

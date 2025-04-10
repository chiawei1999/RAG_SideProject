# -*- coding: utf-8 -*-
"""
向量化模組：將處理後文本做 chunk、嵌入、寫入 Qdrant
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from uuid import uuid4

QDRANT_URL = "http://localhost:6333"  # 或你的遠端 Qdrant URL
COLLECTION_NAME = "rag_docs"
EMBEDDING_DIM = 384  # jina-zh small 模型為 384 維

def split_text(text: str, chunk_size=500, overlap=100):
    """
    文本切割：手刻版本（模擬 RecursiveCharacterTextSplitter）
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_clean_text(path: str) -> str:
    """
    讀取清理後的 txt
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def vectorize_and_store(text_path: str):
    """
    主流程：切割 ➜ 嵌入 ➜ 寫入 Qdrant
    """
    # 初始化 Qdrant
    client = QdrantClient(QDRANT_URL)
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        print(f"✅ 已建立集合：{COLLECTION_NAME}")
    else:
        print(f"📦 使用既有集合：{COLLECTION_NAME}")

    # 載入文本與切割
    raw_text = load_clean_text(text_path)
    chunks = split_text(raw_text)

    # 向量化模型
    model = SentenceTransformer("jinaai/jina-embeddings-v2-small-zh")
    embeddings = model.encode(chunks, normalize_embeddings=True)

    # 組成 Qdrant Point 結構
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embeddings[i],
            payload={"text": chunks[i]}
        )
        for i in range(len(chunks))
    ]

    # 寫入資料庫
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🔐 共上傳 {len(points)} 筆 chunk 至 Qdrant")


if __name__ == "__main__":
    vectorize_and_store("./data/clean_text.txt")

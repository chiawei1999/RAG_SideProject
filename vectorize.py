import os
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from uuid import uuid4

config = dotenv_values(".env")
QDRANT_URL = config["QDRANT_URL"] # Qdrant 伺服器的 URL
COLLECTION_NAME = config["COLLECTION_NAME"] # Qdrant 集合名稱
EMBEDDING_DIM = int(config.get("EMBEDDING_DIM", 384)) # 嵌入向量的維度

def parse_text_blocks(text: str) -> list[dict]:
    """
    解析標記有 【type: xxx】 的文本區塊
    
    Args:
        text: 包含多個標記區塊的文本
        
    Returns:
        list: 包含每個區塊類型和內容的字典列表
    """
    blocks = []
    # 使用 【type: 分割文本
    sections = text.split("【type:")
    
    # 第一個分割可能是空的（如果文本以【type:開頭）
    sections = [s for s in sections if s.strip()]
    
    for section in sections:
        # 提取類型和內容
        try:
            # 分割出類型和內容
            type_end = section.find("】")
            if type_end == -1:
                continue
                
            block_type = section[:type_end].strip()
            content = section[type_end+1:].strip()
            
            blocks.append({
                "type": block_type,
                "content": content
            })
        except Exception as e:
            print(f"解析區塊時出錯: {e}")
            continue
            
    return blocks

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
    client = QdrantClient(
    url=QDRANT_URL,
    api_key=config["QDRANT_API_KEY"]  # 新增這一行，記得在 .env 補上
    )
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

    # 載入文本
    raw_text = load_clean_text(text_path)
    
    # 解析文本區塊
    blocks = parse_text_blocks(raw_text)
    
    # 向量化模型
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-zh")
    
    # 準備所有的 chunks 和對應的 metadata
    all_chunks = []
    all_metadata = []
    
    for i, block in enumerate(blocks):
        block_chunks = split_text(block["content"])
        for chunk in block_chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "type": block["type"],
                "block_id": i,  # 新增欄位
                "text": chunk
            })

    
    print(f"📄 共解析出 {len(blocks)} 個區塊，產生 {len(all_chunks)} 個 chunks")
    
    # 向量化所有 chunks
    embeddings = model.encode(all_chunks, normalize_embeddings=True)
    
    # 組成 Qdrant Point 結構
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=embeddings[i],
            payload=all_metadata[i]
        )
        for i in range(len(all_chunks))
    ]

    # 寫入資料庫
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🔐 共上傳 {len(points)} 筆 chunk 至 Qdrant")


if __name__ == "__main__":
    vectorize_and_store("./data/clean_text.txt")
import os
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchParams
import google.generativeai as genai

# 載入環境變數
config = dotenv_values(".env")
QDRANT_URL = config["QDRANT_URL"]
COLLECTION_NAME = config["COLLECTION_NAME"]
EMBEDDING_DIM = int(config.get("EMBEDDING_DIM", 384))
OPENAI_TOP_K = int(config.get("TOP_K", 3))  # 每次抓取前 K 筆相似內容
GEMINI_API_KEY = config["GEMINI_API_KEY"]

# 初始化 Gemini 模型
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-pro")


class RAGPipeline:
    def __init__(self):
        # 初始化 Qdrant 與模型
        self.client = QdrantClient(QDRANT_URL)
        self.model = SentenceTransformer("jinaai/jina-embeddings-v2-small-zh")

    def search_similar_docs(self, query: str, top_k: int = OPENAI_TOP_K):
        # 向量化 query 並查詢相似內容
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128),  # 可調整查詢精度
        )
        return results

    def build_prompt(self, query: str, search_results: list):
        # 將相似內容組合成提示詞
        context = "\n\n".join([hit.payload["text"] for hit in search_results])
        prompt = f"""
以下是我的履歷內容：
{context}

根據上述內容，請回答以下問題：
{query}
"""
        return prompt

    def query(self, user_input: str) -> str:
        # 主流程：查詢 + prompt + 回答
        results = self.search_similar_docs(user_input)
        prompt = self.build_prompt(user_input, results)
        response = llm.generate_content(prompt)
        return response.text

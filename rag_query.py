import os
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# 載入環境變數
config = dotenv_values(".env")
QDRANT_URL = config.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = config.get("COLLECTION_NAME", "rag_docs")
TOP_K = int(config.get("TOP_K", "3"))
GEMINI_API_KEY = config["GEMINI_API_KEY"]
QDRANT_API_KEY = config.get("QDRANT_API_KEY", None)

# 初始化 Gemini 模型
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-1.5-pro"
llm = genai.GenerativeModel(MODEL_NAME)

class RAGPipeline:
    def __init__(self):
        qdrant_params = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            qdrant_params["api_key"] = QDRANT_API_KEY
        self.client = QdrantClient(**qdrant_params)

        # 載入與向量庫一致的嵌入模型
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def classify_query_type(self, query: str) -> str:
        """
        使用 Gemini 模型判斷問句應對哪一類履歷段落
        """
        prompt = f"""
        以下是履歷的類別：
        1. 基本資料
        2. 學歷
        3. 工作經歷
        4. 技能
        5. 專案
        6. 人格特質

        使用者問了這個問題：
        「{query}」

        請只回覆其中一個類別名稱：基本資料 / 學歷 / 工作經歷 / 技能 / 專案 / 人格特質。
        如果無法分類，請回覆「無法分類」。
        """
        response = llm.generate_content(prompt)
        answer = response.text.strip()
        valid_types = ["基本資料", "學歷", "工作經歷", "技能", "專案", "人格特質"]
        return answer if answer in valid_types else "無法分類"

    def search_similar_docs(self, query: str, top_k: int = TOP_K, filter_type: str = None):
        """
        使用 Qdrant 查詢相似內容，可指定分類條件
        """
        query_vector = self.model.encode(query).tolist()
        params = {
            "collection_name": COLLECTION_NAME,
            "query_vector": query_vector,
            "limit": top_k
        }
        if filter_type:
            params["query_filter"] = {
                "must": [{"key": "type", "match": {"value": filter_type}}]
            }
        return self.client.search(**params)

    def build_prompt(self, query: str, results: list) -> str:
        """
        建立 LLM 回答用的 prompt，並組合 context
        """
        context = "\n\n".join(
            f"【{r.payload['type']}】\n{r.payload['text']}"
            for r in results if hasattr(r, 'payload') and 'type' in r.payload and 'text' in r.payload
        ) or "未能找到相關履歷內容。"

        return f"""
        你是一位人資專員，正在閱讀劉家瑋的履歷，請針對他的經歷內容回答問題。
        以下是劉家瑋的履歷內容：
        {context}

        請根據上述內容，回答以下問題：
        {query}

        規則如下：
        1. 回答內容請大概引用履歷中的原文，不要過多推論或補充額外知識。
        2. 回答時請將「我」統一改為「家瑋」。
        3. 若查無相關資訊，請回答「查無相關資料」。
        """

    def query(self, user_input: str) -> str:
        """
        主流程：分類 ➜ 查詢 ➜ 組 Prompt ➜ 呼叫 LLM
        """
        query_type = self.classify_query_type(user_input)
        filter_type = query_type if query_type != "無法分類" else None
        results = self.search_similar_docs(user_input, filter_type=filter_type)
        if not results:
            return "抱歉，我找不到相關的資訊來回答您的問題。"
        prompt = self.build_prompt(user_input, results)
        return llm.generate_content(prompt).text

if __name__ == "__main__":
    rag = RAGPipeline()
    print(rag.query("請問家瑋的個性如何？"))

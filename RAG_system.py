# 安裝必要的套件
# pip install langchain-openai langchain-community langchain qdrant-client sentence-transformers python-dotenv

import os
from typing import List
from dotenv import dotenv_values

# 向量化相關
from sentence_transformers import SentenceTransformer

# Qdrant 向量資料庫
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams

# LangChain 組件
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 載入環境變數
config = dotenv_values(".env")

# 配置變數
QDRANT_URL = config.get("QDRANT_URL")  # Qdrant 雲服務地址
QDRANT_API_KEY = config.get("QDRANT_API_KEY")  # Qdrant API 金鑰
COLLECTION_NAME = config.get("COLLECTION_NAME")  # 集合名稱
EMBEDDING_DIM = int(config.get("EMBEDDING_DIM", "768"))  # 使用的嵌入模型維度

# OpenAI API 金鑰 (或其他語言模型 API)
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

# 使用 Jina 處理文件並向量化
def process_documents(file_path: str):
    """
    讀取並處理文件
    """
    # 載入文件
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    # 分割文件為較小的塊
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    return splits

# 初始化 Qdrant 客戶端並創建集合
def init_qdrant():
    """
    初始化 Qdrant 向量資料庫
    """
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    
    # 檢查集合是否存在，不存在則創建
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
        )
        print(f"已創建集合: {COLLECTION_NAME}")
    else:
        print(f"使用現有集合: {COLLECTION_NAME}")
    
    return client

# 設置嵌入模型
def get_embeddings():
    """
    設置文本嵌入模型
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 較小的模型，適合初始測試
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

# 建立 RAG 系統
def build_rag_system(query: str, file_path: str):
    """
    構建並使用 RAG 系統
    """
    # 初始化 Qdrant
    client = init_qdrant()
    
    # 設置嵌入模型
    embeddings = get_embeddings()
    
    # 處理並加載文件到向量資料庫
    splits = process_documents(file_path)
    
    # 將處理後的文件存入 Qdrant
    vector_store = Qdrant.from_documents(
        documents=splits,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    # 設置語言模型
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # 設置提示模板
    template = """使用以下信息來回答問題。如果你不知道答案，就說你不知道，不要嘗試編造答案。
    
    背景信息:
    {context}
    
    問題: {question}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 創建 RAG 鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 執行查詢
    result = qa_chain.invoke(query)
    
    return result

# 主函數
def main():
    # 示例文件路徑
    file_path = "interview.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    # 使用者查詢
    query = "面試時要注意什麼?"
    
    # 執行 RAG 查詢
    response = build_rag_system(query, file_path)
    print(f"問題: {query}")
    print(f"回答: {response}")

# 執行示例
if __name__ == "__main__":
    main()
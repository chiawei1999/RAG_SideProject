# RAG 問答系統專案

本專案旨在實作一套從文本前處理到語意查詢與生成式回答的完整 RAG（Retrieval-Augmented Generation）架構，使用者可針對輸入問句，從自定義知識庫中擷取語意相關內容並生成精確回覆。

## 📁 專案結構

- `text_preprocess.py`：原始文本清理，保留格式並去除多餘符號。
- `vectorize.py`：清理文本切割後進行向量化，儲存至 Qdrant 向量資料庫。
- `rag_query.py`：整合查詢與生成，透過 Gemini API 回答用戶問題。
- `raw_text.txt`：原始輸入文本，需標記【type: xxx】格式。
- `clean_text.txt`：經前處理後的純淨文本，可供向量化使用。

## 🔧 使用技術

- **向量資料庫**：Qdrant
- **向量模型**：
  - 嵌入階段使用 `jinaai/jina-embeddings-v2-base-zh`
  - 查詢階段使用 `all-mpnet-base-v2`
- **生成模型**：Gemini 1.5 Pro (Google Generative AI)
- **後端邏輯**：Python（包含 dotenv、sentence-transformers 等）

## 🔄 RAG 流程

1. **前處理**：清理 `raw_text.txt` ➜ 產出 `clean_text.txt`
2. **向量化**：載入清理後文本，切割並轉為向量，寫入 Qdrant
3. **查詢流程**：
   - 接收使用者問題
   - 查詢 Qdrant 中語意相似的段落
   - 組合 context 並建立 prompt 給 Gemini 回答
   - 回傳回覆內容

## ✅ 專案特色

- 支援個人化履歷知識庫查詢
- 使用兩階段向量模型以提升查詢準確率
- 提供簡單易懂的模組化程式結構，方便維護與擴充

## 📎 開發者

劉家瑋  
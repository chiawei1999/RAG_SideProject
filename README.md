# RAG 系統實作文件

## 專案概述

本專案實現了一個基於檢索增強生成（Retrieval-Augmented Generation, RAG）的問答系統，能夠針對特定文件內容進行問答。系統使用向量資料庫存儲文件的語義表示，並結合大型語言模型提供精確的回答。

## 功能特點

- 文件處理與分塊：自動將長文件分割成適當大小的文本塊
- 向量化存儲：使用高效的語義嵌入模型將文本轉換為向量
- 相似度檢索：基於用戶問題快速檢索相關文本
- 智能回答生成：結合檢索結果與大型語言模型生成高質量回答

## 技術架構

- **嵌入模型**：HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **向量資料庫**：Qdrant
- **語言模型**：OpenAI GPT-3.5-turbo
- **框架**：LangChain

## 環境設置

### 必要套件安裝

```bash
pip install langchain-openai langchain-community langchain qdrant-client sentence-transformers python-dotenv
```

### 環境變數配置

在專案根目錄創建 `.env` 檔案，並設置以下參數：

```env
QDRANT_URL=你的Qdrant服務URL
QDRANT_API_KEY=你的Qdrant API金鑰
COLLECTION_NAME=你想使用的集合名稱
EMBEDDING_DIM=768
OPENAI_API_KEY=你的OpenAI API金鑰
```

## 使用方法

### 基本使用

1. 確保已安裝所有必要套件
2. 配置好環境變數
3. 準備好要進行問答的文本檔案
4. 運行程式並輸入問題

```python
from RAG_system import build_rag_system

# 文件路徑
file_path = "interview.txt"

# 使用者問題
question = "面試前應該如何準備？"

# 獲取回答
response = build_rag_system(question, file_path)
print(f"問題: {question}")
print(f"回答: {response}")
```

## 系統流程

1. **文件處理**：將文件載入並分割成較小的文本塊
2. **向量化**：使用嵌入模型將文本塊轉換為向量表示
3. **存儲**：將向量和原始文本存儲在Qdrant資料庫中
4. **檢索**：當收到問題時，將問題向量化並在資料庫中檢索最相關的文本塊
5. **回答生成**：將檢索到的文本塊作為上下文，結合用戶問題，使用語言模型生成最終回答

## 示例應用

本系統目前配置用於處理面試相關問題，基於「interview.txt」文件中的面試準備和技巧內容。您可以輸入有關面試準備、注意事項等問題獲取回答。

## 系統擴展

### 支援更多文件類型

目前系統僅支援純文本檔案，若要支援更多檔案類型，可以替換`TextLoader`為其他LangChain支援的文檔加載器，如：

- PDFLoader：處理PDF文件
- CSVLoader：處理CSV文件
- DocxLoader：處理Word文件

### 自訂嵌入模型

若需要更高效或更適合中文的嵌入模型，可以修改`get_embeddings`函數，替換為其他支援的模型。

## 故障排除

- **記憶體錯誤**：處理大型文件時，可能需要調整`chunk_size`參數減小分塊大小
- **向量維度錯誤**：確保`EMBEDDING_DIM`環境變數與所選嵌入模型的維度一致
- **連接錯誤**：確保Qdrant服務URL和API金鑰正確，且網絡連接正常
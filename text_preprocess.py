import os
import re

def clean_text(text: str) -> str:
    """
    溫和清理：保留換行與格式，只移除 HTML tag 與多餘空行
    """
    # 移除 HTML tag（如有）
    text = re.sub(r"<[^>]+>", "", text)

    # 移除多餘空行（保留單一換行）
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 每行首尾去空白
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()



def process_raw_txt(input_path: str, output_path: str):
    """
    讀取原始文本，清理後儲存為新的檔案
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✔ 文本處理完成，儲存至：{output_path}")


if __name__ == "__main__":
    # 原始 txt 與清理後 txt 的路徑
    input_path = "./data/raw_text.txt"
    output_path = "./data/clean_text.txt"
    process_raw_txt(input_path, output_path)

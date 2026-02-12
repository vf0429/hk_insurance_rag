
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ================= 配置部分 =================
DATA_PATH = "./data"  # Changed from "./one_degree" to the root data directory
DB_PATH = "./chroma_db"                  # 向量库存储路径

# 嵌入模型配置 (必须与 main.py 一致)
# Using the key and base URL from main.py context
DEEPSEEK_API_KEY = "sk-ufsptsmrlhiphmszabdpufeqqvvnyjmeudmtnrlfgpfzlyma"
DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"

EMBEDDING_MODEL = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_base=DEEPSEEK_BASE_URL,
    openai_api_key=DEEPSEEK_API_KEY,
    check_embedding_ctx_length=False,
    chunk_size=32 # CRITICAL: Added back to prevent 413 error
)

# ================= 核心处理函数 =================

def load_and_process_documents(data_root):
    all_processed_docs = []
    
    # 1. 定义 Markdown 标题切分器 (结构化切分)
    headers_to_split_on = [
        ("#", "Section_Name"),
        ("##", "Chapter_Name"),
        ("###", "Clause_Name"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=True
    )

    # 2. 定义文本递归切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    if not os.path.exists(data_root):
        print(f"Directory {data_root} does not exist!")
        return []

    # Recursive scan using os.walk
    for root, dirs, files in os.walk(data_root):
        # Skip hidden folders (but not the current/parent directory indicators)
        parts = root.split(os.sep)
        if any(part.startswith('.') and part not in ['.', '..'] for part in parts):
            continue

        # Extract provider: the first subfolder under data/
        # e.g., if data_root is ./data and root is ./data/bluecross/sub, provider is bluecross
        relative_path = os.path.relpath(root, data_root)
        if relative_path == ".":
            continue # Skip files in the root data folder
        
        provider_name = relative_path.split(os.sep)[0]

        print(f"\n--- 正在处理保险商: {provider_name} (当前路径: {root}) ---")
        
        for filename in files:
            if not filename.endswith(".md"):
                continue
                
            file_path = os.path.join(root, filename)
            print(f"  正在處理文件: {filename}...")
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # --- 特殊處理 A: 價格表/對比表 (不切分) ---
            if "plans_pricing" in filename:
                from langchain_core.documents import Document
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "provider": provider_name,
                        "language": "zh" if "_zh" in filename else "en",
                        "type": "Pricing Table"
                    }
                )
                all_processed_docs.append(doc)
                continue

            # --- 常規處理 B: 保單條款 (切分) ---
            
            # 步骤 1: 按 Header 切分
            md_docs = markdown_splitter.split_text(text)
            
            # 步骤 2: 注入 Metadata 标签
            for doc in md_docs:
                doc.metadata["language"] = "zh" if "_zh" in filename else "en"
                doc.metadata["source"] = filename
                doc.metadata["provider"] = provider_name
                
            # 步骤 3: 再次切分
            final_splits = text_splitter.split_documents(md_docs)
            all_processed_docs.extend(final_splits)

    return all_processed_docs

# ================= 主执行逻辑 =================

if __name__ == "__main__":
    # 1. 初始化向量库 (不清理舊數據)
    print(f"正在初始化/加载向量库: {DB_PATH}")
    
    # Initialize Chroma first if it doesn't exist
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=EMBEDDING_MODEL
    )

    # 2. 加载与处理新文档
    docs = load_and_process_documents(DATA_PATH)
    print(f"共生成 {len(docs)} 個切片。")

    if not docs:
        print("No documents found. Exiting.")
        exit(0)

    # 3. 向量化入庫 (使用穩定 ID 實現 upsert)
    print("正在生成向量並更新向量庫 (使用穩定 ID 以防止重複)...")
    
    # Generate stable IDs based on filename and content/index to allow upserting
    import hashlib
    def generate_id(doc, index):
        # ID is a hash of (provider + filename + chunk index)
        identifier = f"{doc.metadata.get('provider')}_{doc.metadata.get('source')}_{index}"
        return hashlib.md_path(identifier.encode()).hexdigest() if hasattr(hashlib, 'md_path') else hashlib.md5(identifier.encode()).hexdigest()

    ids = [generate_id(doc, i) for i, doc in enumerate(docs)]
    
    # add_documents will add or update if IDs are provided
    vectorstore.add_documents(documents=docs, ids=ids)
    
    print("✅ 數據更新完成！(已實現 Upsert)")

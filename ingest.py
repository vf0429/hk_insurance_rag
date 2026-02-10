
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ================= 配置部分 =================
DATA_PATH = "./one_degree"  # Changed from "./data/insurance_provider" to match actual path
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

def load_and_process_documents(directory):
    all_processed_docs = []
    
    # 1. 定义 Markdown 标题切分器 (结构化切分)
    # 这会把 #, ##, ### 的内容提取到 metadata 中
    headers_to_split_on = [
        ("#", "Section_Name"),
        ("##", "Chapter_Name"),
        ("###", "Clause_Name"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=True # 标题会被移到 metadata，我们在后面会拼回去
    )

    # 2. 定义文本递归切分器 (防止单节过长)
    # 【关键修改】：chunk_size 设为 2000，确保 Section 1.1 这种长逻辑不断开
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist!")
        return []

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue
            
        file_path = os.path.join(directory, filename)
        print(f"正在处理: {filename}...")
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # --- 特殊处理 A: 价格表/对比表 (不切分) ---
        if "plans_pricing" in filename:
            from langchain_core.documents import Document
            # 表格文件直接作为一整块，保证 LLM 能读懂行和列
            doc = Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "language": "zh" if "_zh" in filename else "en",
                    "type": "Pricing Table"
                }
            )
            all_processed_docs.append(doc)
            continue

        # --- 常规处理 B: 保单条款 (切分) ---
        
        # 步骤 1: 按 Header 切分 (获取结构信息)
        md_docs = markdown_splitter.split_text(text)
        
        # 步骤 2: 注入 Metadata 标签 (给切片“贴名牌”)
        # 这是修复 DeepSeek "看不见标题" 问题的关键！
        for doc in md_docs:
            # 自动判断语言
            doc.metadata["language"] = "zh" if "_zh" in filename else "en"
            doc.metadata["source_file"] = filename
            
            # 构建面包屑导航: [Section 1 > 1.1 Covered Conditions]
            breadcrumbs = []
            if "Section_Name" in doc.metadata: breadcrumbs.append(doc.metadata["Section_Name"])
            if "Chapter_Name" in doc.metadata: breadcrumbs.append(doc.metadata["Chapter_Name"])
            if "Clause_Name" in doc.metadata: breadcrumbs.append(doc.metadata["Clause_Name"])
            
            source_tag = " > ".join(breadcrumbs)
            # 【核心黑魔法】：把标题硬拼回正文开头
            # 这样检索时，LLM 看到的第一句话就是 "**SOURCE: ...**"
            doc.page_content = f"**SOURCE: [{source_tag}]**\n\n{doc.page_content}"

        # 步骤 3: 再次切分 (处理超长段落，但因为 chunk_size=2000，大部分不会被切碎)
        final_splits = text_splitter.split_documents(md_docs)
        all_processed_docs.extend(final_splits)

    return all_processed_docs

# ================= 主执行逻辑 =================

if __name__ == "__main__":
    # 1. 清理旧数据库 (可选，防止重复)
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        print("已清理旧向量库...")

    # 2. 加载与处理
    docs = load_and_process_documents(DATA_PATH)
    print(f"共生成 {len(docs)} 个切片。")

    if not docs:
        print("No documents found. Exiting.")
        exit(0)

    # 3. 向量化入库
    print("正在生成向量并入库 (使用 BAAI/bge-m3)...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=EMBEDDING_MODEL,
        persist_directory=DB_PATH
    )
    
    print("✅ 入库完成！请运行 main.py 进行测试。")

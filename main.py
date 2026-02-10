
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
import re

def clean_deepseek_output(raw_text):
    # Log raw output for debugging
    logger.info(f"Raw LLM Output (first 100 chars): {raw_text[:100]}...")
    
    # 1. Remove complete <think>...</think> block
    cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    
    # 2. Handle missing start tag (common issue): remove everything before </think>
    if '</think>' in cleaned_text:
        cleaned_text = re.sub(r'^.*?</think>', '', cleaned_text, flags=re.DOTALL)
    
    # 3. Cleanup whitespace
    return cleaned_text.strip()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEEPSEEK_API_KEY = "sk-ufsptsmrlhiphmszabdpufeqqvvnyjmeudmtnrlfgpfzlyma" # Provided by user on 2026-02-10
DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"

app = FastAPI(title="OneDegree Pet Insurance RAG API")

@app.get("/")
async def root():
    return {"message": "Pet Insurance RAG API is running. use /ask to query."}

# --- Global Components ---
embeddings = None
vectorstore = None
retriever = None
rag_chain = None

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstore, retriever, rag_chain
    
    logger.info("Initializing Embeddings (BAAI/bge-m3)...")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=32
    )
    
    logger.info("Loading Vector Store...")
    if not os.path.exists("./chroma_db"):
        logger.error("Chroma DB not found! Run ingest.py first.")
        raise RuntimeError("Chroma DB not found.")
        
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    logger.info("Initializing MiniMAX LLM...")
    llm = ChatOpenAI(
        model="Pro/MiniMaxAI/MiniMax-M2.1",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        max_tokens=1024,
        temperature=0.1
    )
    
    template = """
    ### Role
    You are "Insurance Policy Assistant", a helpful and professional AI customer support agent. 

    ### INTERNAL LOGIC CHECK (Mental Sandbox - DO NOT OUTPUT THIS SECTION)
    Before answering, you MUST privately verify:
    1. **User Scenario**: New Policy vs. Upgrade? (If New, IGNORE Section 4.6).
    2. **Waiting Period**: Is there a 28-day waiting period for accidents? (Section 2.1).
    3. **Age Exceptions**: Is the pet 5+ years old? Check Section 1.1 "Special Coverage Rules" (The One-Year Rule).
    4. **General Exceptions**: Are there specific exclusions (e.g., Breeding, dental)?

    ### OUTPUT RULES
    - **Language**: Answer in the SAME language as the User's Question (Traditional Chinese or English).
    - **Thinking Process**: You may think in any language, but the FINAL output must match the user's language.
    - **Tone**: Empathetic, clear, and direct. Avoid legal jargon where possible.
    - **Length**: Keep it concise (under 200 words).
    - **Structure**:
        1. **Direct Answer**: Start with a clear "Yes", "No", or "Conditional" conclusion.
        2. **The "Why"**: Explain the rule simply (e.g., "Because there is a 28-day waiting period...").
        3. **Key Exception (If applicable)**: Only mention exceptions if they apply to the user (e.g., the One-Year Rule for 5yo pets).
        4. **Action**: Suggest the next step (e.g., "Check your policy schedule").

    ### Context
    {context}

    ### User Question
    {question}

    ### Your Response
    """



    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([f"[Section: {d.metadata.get('Clause_Name', d.metadata.get('Section_Name', 'Unknown'))}]\n{d.page_content}" for d in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG Chain initialized.")

# --- API Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_insurance_policy(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    try:
        logger.info(f"Received query: {request.query}")
        
        # 1. Retrieve docs for sources
        docs = await retriever.ainvoke(request.query)
        sources = list(set([d.metadata.get("Clause_Name", d.metadata.get("Section_Name", "Policy Doc")) for d in docs]))
        
        # 2. Generate answer
        raw_answer = await rag_chain.ainvoke(request.query)
        logger.info(f"Raw LLM Response Body: {repr(raw_answer)}") # Force full log
        answer = clean_deepseek_output(raw_answer)
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

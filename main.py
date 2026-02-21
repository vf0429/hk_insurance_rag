
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

app = FastAPI(title="Petwell Insurance RAG API")

@app.get("/")
async def root():
    return {"message": "Petwell Pet Insurance RAG API is running. use /ask to query."}

# --- Global Components ---
embeddings = None
vectorstore = None
# retriever = None # We will create retriever dynamically or use a base one
rag_chain = None
llm = None
prompt = None

@app.on_event("startup")
async def startup_event():
    global embeddings, vectorstore, rag_chain, llm, prompt
    
    logger.info("Initializing Embeddings (BAAI/bge-m3)...")
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
    
    logger.info("Initializing MiniMAX LLM...")
    llm = ChatOpenAI(
        model="Pro/MiniMaxAI/MiniMax-M2.1",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        max_tokens=1024,
        temperature=0.1
    )
    
    template =''' 
        ### Role
        You are the "Petwell AI Specialist," a professional expert on the Hong Kong pet insurance market. You provide answers based on retrieved policy documents from different providers (e.g., Blue Cross, OneDegree).

        ### INTERNAL LOGIC (Mental Sandbox - DO NOT OUTPUT)
        1. **Identify Provider**: Which company is the user asking about? If unspecified, check all available context.
        2. **Eligibility Check**: Verify pet age (e.g., 6 months to 8 years for Blue Cross) and breed (check for excluded breeds like Pit Bulls).
        3. **Wait Times & Limits**: Apply provider-specific timelines (e.g., 90-day cancer wait for Blue Cross) and HK$ benefit limits.
        4. **Conflict Resolution**: If policies differ, clearly state the terms for each provider separately.

        ### OUTPUT RULES
        - **No Thinking Process**: Do NOT output "Internal Logic," "Thinking," or "Sandbox." Start the response immediately.
        - **Language**: Match the user's language (Traditional Chinese or English).
        - **Formatting**: Use **Bold Text** for all dollar amounts (e.g., **HK$1,000**) and timeframes (e.g., **30 days**).
        - **Structure**:
            1. **Direct Answer**: A clear "Yes/No" or summary.
            2. **Details**: Specify the rule/limit and name the provider (e.g., "Under Blue Cross Plan B...").
            3. **Comparison (If relevant)**: Briefly note if another provider has a different rule.
            4. **Next Step**: One actionable suggestion.

        ### Context
        {context}

        ### User Question
        {question}

        ### Your Response
        '''


    prompt = ChatPromptTemplate.from_template(template)
    logger.info("RAG Chain initialized (components loaded).")

# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    provider: str = None  # Optional provider filter

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

def format_docs(docs):
    # Goal: '--- SOURCE: [provider] ([filename]) ---'
    formatted_chunks = []
    for d in docs:
        provider = d.metadata.get('provider', 'Unknown')
        filename = d.metadata.get('source', 'Unknown')
        header = f"--- SOURCE: {provider} ({filename}) ---"
        formatted_chunks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(formatted_chunks)

@app.post("/ask", response_model=QueryResponse)
async def ask_insurance_policy(request: QueryRequest):
    if not vectorstore:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    try:
        logger.info(f"Received query: {request.query} (Filter: {request.provider})")
        
        # 1. Prepare retriever with optional filter
        search_kwargs = {"k": 6}
        
        # Detect provider in query if not explicitly provided
        effective_provider = request.provider
        # Ignore default "string" value or empty values
        if effective_provider in [None, "", "string"]:
            effective_provider = None
            
        if not effective_provider:
            query_lower = request.query.lower()
            if "blue cross" in query_lower or "bluecross" in query_lower or "藍十字" in query_lower:
                effective_provider = "bluecross"
            elif "one degree" in query_lower or "onedegree" in query_lower:
                effective_provider = "one_degree"
            elif "prudential" in query_lower or "pruchoice" in query_lower or "保誠" in query_lower:
                effective_provider = "prudential"
            elif "bolttech" in query_lower:
                effective_provider = "bolttech"

        if effective_provider and effective_provider != "string":
            logger.info(f"Active filter: {effective_provider}")
            search_kwargs["filter"] = {"provider": effective_provider}
            
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs=search_kwargs
        )
        
        # 2. Retrieve documents
        docs = await retriever.ainvoke(request.query)
        sources = list(set([f"{d.metadata.get('provider', 'Unknown')} ({d.metadata.get('source', 'Unknown')})" for d in docs]))
        
        # 3. Build and run chain manually to handle dynamic context
        context_str = format_docs(docs)
        
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        
        raw_answer = await chain.ainvoke({"context": context_str, "question": request.query})
        logger.info(f"Raw LLM Response Body: {repr(raw_answer)}")
        answer = clean_deepseek_output(raw_answer)
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


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
from typing import Optional

# --- Provider Display Names ---
PROVIDER_DISPLAY_NAMES = {
    "bluecross": "Blue Cross 藍十字",
    "one_degree": "OneDegree",
    "prudential": "Prudential 保誠",
    "bolttech": "Bolttech",
}

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

# --- GET /providers ---
@app.get("/providers")
async def list_providers():
    """Return list of available insurance providers from ChromaDB."""
    if not vectorstore:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        results = vectorstore.get(include=[])
        # Get distinct providers from metadata
        all_meta = vectorstore.get(include=["metadatas"])
        providers_set = set()
        for m in all_meta["metadatas"]:
            p = m.get("provider")
            if p:
                providers_set.add(p)
        
        provider_list = [
            {"id": pid, "name": PROVIDER_DISPLAY_NAMES.get(pid, pid)}
            for pid in sorted(providers_set)
        ]
        return {"providers": provider_list}
    except Exception as e:
        logger.error(f"Error listing providers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        You are the "Petwell AI Specialist," a professional expert on the Hong Kong pet insurance market. You provide answers based on retrieved policy documents from different providers (e.g., Blue Cross, OneDegree, Prudential).

        ### Provider Context
        The user is currently viewing: {active_provider_name}
        - If a specific provider is active (not "All Providers"), focus your answer on that provider's policy.
        - If "All Providers" is active and the question is provider-specific but no provider is mentioned, note which providers you have information for and suggest the user select one.
        - If the question is general, compare across all available providers in the context.

        ### INTERNAL LOGIC (Mental Sandbox - DO NOT OUTPUT)
        1. **Identify Provider**: Which company is the user asking about? Use the active provider context above.
        2. **Eligibility Check**: Verify pet age and breed against the active provider's rules.
        3. **Wait Times & Limits**: Apply provider-specific timelines and HK$ benefit limits.
        4. **Conflict Resolution**: If policies differ, clearly state the terms for each provider separately.

        ### OUTPUT RULES
        - **No Thinking Process**: Do NOT output "Internal Logic," "Thinking," or "Sandbox." Start the response immediately.
        - **Language**: You MUST respond in the SAME language the user used in their question. If the user writes in English, you MUST reply entirely in English. If the user writes in Traditional Chinese, reply in Traditional Chinese. NEVER switch languages. The language of the source documents does NOT matter — always match the user's language.
        - **Formatting**: Use **Bold Text** for all dollar amounts (e.g., **HK$1,000**) and timeframes (e.g., **30 days**).
        - **Structure**:
            1. **Direct Answer**: A clear "Yes/No" or summary.
            2. **Details**: Specify the rule/limit and name the provider (e.g., "Under Blue Cross Plan B...").
            3. **Comparison (If relevant)**: Briefly note if another provider has a different rule.
            4. **Next Step**: One actionable suggestion.

        ### Conversation History (for context only — do NOT reference or mention this section in your response)
        {chat_history}

        ### Retrieved Policy Documents
        {context}

        ### User Question
        {question}

        ### Your Response
        '''


    prompt = ChatPromptTemplate.from_template(template)
    logger.info("RAG Chain initialized (components loaded).")

# --- API Models ---
class ChatTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    query: str
    provider: Optional[str] = None        # Explicit provider filter (from backend)
    session_id: Optional[str] = None      # Session ID (passed through from backend)
    chat_history: Optional[list[ChatTurn]] = None  # Last N conversation turns

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    active_provider: Optional[str] = None   # Which provider was used for filtering
    session_id: Optional[str] = None        # Echo back for frontend tracking

def format_docs(docs):
    """Format retrieved documents with source headers."""
    formatted_chunks = []
    for d in docs:
        provider = d.metadata.get('provider', 'Unknown')
        filename = d.metadata.get('source', 'Unknown')
        header = f"--- SOURCE: {provider} ({filename}) ---"
        formatted_chunks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(formatted_chunks)

def format_chat_history(chat_history: list[ChatTurn], max_turns: int = 5) -> str:
    """Format the last N conversation turns for prompt injection.
    Truncates assistant answers to 300 chars to save context window."""
    if not chat_history:
        return "None"
    
    # Take only the last max_turns pairs
    recent = chat_history[-(max_turns * 2):]
    
    lines = []
    for turn in recent:
        content = turn.content
        if turn.role == "assistant" and len(content) > 300:
            content = content[:300] + "..."
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    
    return "\n".join(lines)

@app.post("/ask", response_model=QueryResponse)
async def ask_insurance_policy(request: QueryRequest):
    if not vectorstore:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    try:
        logger.info(f"Received query: {request.query} (Filter: {request.provider}, Session: {request.session_id})")
        
        # 1. Prepare retriever with optional filter
        search_kwargs = {"k": 6}
        
        # Resolve effective provider from request
        effective_provider = request.provider
        # Ignore default "string" value or empty values
        if effective_provider in [None, "", "string"]:
            effective_provider = None
            
        # Fallback: detect provider from query keywords
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
        
        # 3. Build prompt inputs
        context_str = format_docs(docs)
        chat_history_str = format_chat_history(request.chat_history) if request.chat_history else "None"
        active_provider_name = PROVIDER_DISPLAY_NAMES.get(effective_provider, "All Providers") if effective_provider else "All Providers"
        
        logger.info(f"Provider: {effective_provider} ({active_provider_name}), History turns: {len(request.chat_history) if request.chat_history else 0}")
        
        # 4. Build and run chain
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        
        raw_answer = await chain.ainvoke({
            "context": context_str,
            "question": request.query,
            "chat_history": chat_history_str,
            "active_provider_name": active_provider_name,
        })
        logger.info(f"Raw LLM Response Body: {repr(raw_answer)}")
        answer = clean_deepseek_output(raw_answer)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            active_provider=effective_provider,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

# embeddings-v3 & reranker-v3 合并
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from modelscope import AutoModel
from sentence_transformers import SentenceTransformer

app = FastAPI(
    title="Jina Reranker 和 Embedding 合并部署服务器"
)

# --- 统一设置 device ---
device = 'cuda:0'

# 2. 加载【两个】模型 (使用不同变量名)

# --- Reranker 模型加载 ---
print("开始加载 Reranker 模型...")
rerank_model_name = r'E:\project\jinaai\jina-reranker-v3\model'
# 关键：变量重命名为 rerank_model
rerank_model = AutoModel.from_pretrained(
    rerank_model_name,
    dtype="auto",
    trust_remote_code=True
).to(device)
rerank_model.eval()
print("Reranker 模型加载完毕.")


# --- Embedding 模型加载 ---
print("开始加载 Embedding 模型...")
embd_model_name = 'jinaai/jina-embeddings-v3'
# 关键：变量重命名为 embd_model
embd_model = SentenceTransformer(
    embd_model_name,
    trust_remote_code=True
).to(device)
embd_model.eval()
print("Embedding 模型加载完毕.")


# 3. Reranker API (/rerank)

# --- Reranker 的 Pydantic 模型 ---
class RerankRequest(BaseModel):
    model: Optional[str] = "jina-reranker-v3"
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False
    return_embeddings: Optional[bool] = False

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None
    embedding: Optional[List[float]] = None

# 关键：重命名 UsageObject
class RerankUsageObject(BaseModel):
    total_tokens: int

class RerankResponse(BaseModel):
    model: str
    usage: RerankUsageObject # 使用重命名后的 Usage
    results: List[RerankResult]

# --- Reranker 的 API 接口 ---
@app.post("/rerank", response_model=RerankResponse)
async def handle_rerank(request: RerankRequest):

    results_list = rerank_model.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        return_embeddings=request.return_embeddings
    )

    processed_results = []
    for res in results_list:
        result_data = {
            "index": res["index"],
            "relevance_score": res["relevance_score"]
        }
        
        if request.return_documents:
            result_data["document"] = res["document"]
        
        if request.return_embeddings and "embedding" in res:
            embedding_data = res["embedding"]
            if hasattr(embedding_data, 'tolist'):
                result_data["embedding"] = embedding_data.tolist()
            else:
                result_data["embedding"] = embedding_data

        processed_results.append(result_data)

    try:
        query_tokens = len(rerank_model.tokenizer.encode(request.query))
        doc_tokens = sum(len(rerank_model.tokenizer.encode(doc)) for doc in request.documents)
        total_tokens = query_tokens + doc_tokens
    except Exception:
        total_tokens = 0

    return {
        "model": request.model or rerank_model_name, 
        "usage": {
            "total_tokens": total_tokens
        },
        "results": processed_results
    }



# 4. Embedding API (/embd)

# --- Embedding 的 Pydantic 模型 ---
class EmbdApiRequest(BaseModel):
    input: List[str]
    model: Optional[str] = None
    task: Optional[str] = None
    dimensions : Optional[int] = None
    embedding_type : Optional[str] = None
    truncate: Optional[bool] = None
    late_chunking: Optional[bool] = None

class EmbdEmbeddingObject(BaseModel):
    object: str
    index: int
    embedding: List[float]

class EmbdUsageObject(BaseModel):
    total_tokens: int
    prompt_tokens: int

class EmbdApiResponse(BaseModel):
    model: str
    object: str
    usage: EmbdUsageObject # 使用重命名后的 Usage
    data: List[EmbdEmbeddingObject]

# --- Embedding 的 API 接口 ---
@app.post("/embd", response_model=EmbdApiResponse)
async def handle_embedding(request: EmbdApiRequest):
    texts_to_process = request.input
    
    # 使用 embd_model
    embeddings_np = embd_model.encode(texts_to_process)
    embeddings_list = embeddings_np.tolist()

    data_list = []
    for i, vector in enumerate(embeddings_list):
        data_list.append(EmbdEmbeddingObject(
            object="embedding",
            index=i,
            embedding=vector
        ))

    total_tokens = sum(len(text) for text in texts_to_process) 

    return {
        "model": embd_model_name, # 使用 embd_model_name
        "object": "list",
        "data": data_list,
        "usage": {
            "total_tokens": total_tokens,
            "prompt_tokens": total_tokens
        }
    }
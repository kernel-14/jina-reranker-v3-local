import torch 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from modelscope import AutoModel

app = FastAPI(
    title = "Jina-reranker-v3 本地部署服务器"
)

device = 'cuda:0'
model_name = r'E:\project\jinaai\jina-reranker-v3\model'
model = AutoModel.from_pretrained(model_name,dtype="auto",trust_remote_code=True).to(device)
model.eval()  # 评估模式，不参与训练

# define request & response
class RerankRequest(BaseModel):
    model: Optional[str] = "jina-reranker-v3"
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False # 默认不返回原始文档，只返回分数
    return_embeddings: Optional[bool] = False # 默认不返回 embeddings

class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None
    embedding: Optional[List[float]] = None

class UsageObject(BaseModel):
    total_tokens: int
    # reranker API 通常只返回 total_tokens
    # prompt_tokens: int 

class RerankResponse(BaseModel):
    model: str
    usage: UsageObject
    results: List[RerankResult]
# api接口
@app.post("/rerank",response_model=RerankResponse)

async def get_embeddings(request:RerankRequest):

    # 执行 rerank
    # model.rerank 会返回一个字典列表，例如:
    # [{'index': 1, 'relevance_score': 0.93, 'document': 'doc text', 'embedding': [...]}, ...]
    results_list = model.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        return_embeddings=request.return_embeddings
    )

    # 处理返回结果，使其匹配 API 格式
    processed_results = []
    for res in results_list:
        result_data = {
            "index": res["index"],
            "relevance_score": res["relevance_score"]
        }
        
        # 根据请求参数，决定是否包含 document 和 embedding
        if request.return_documents:
            result_data["document"] = res["document"]
        
        if request.return_embeddings and "embedding" in res:
            # 确保 embedding 是列表 (numpy array -> list)
            embedding_data = res["embedding"]
            if hasattr(embedding_data, 'tolist'):
                result_data["embedding"] = embedding_data.tolist()
            else:
                result_data["embedding"] = embedding_data

        processed_results.append(result_data)

    # 计算 token 使用量
    try:
        query_tokens = len(model.tokenizer.encode(request.query))
        doc_tokens = sum(len(model.tokenizer.encode(doc)) for doc in request.documents)
        total_tokens = query_tokens + doc_tokens
    except Exception:
        total_tokens = 0 # 如果计算失败，则返回 0

    # 4. 构建最终响应
    return {
        "model": request.model or model_name,
        "usage": {
            "total_tokens": total_tokens
        },
        "results": processed_results
    }
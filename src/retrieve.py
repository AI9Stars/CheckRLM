from FlagEmbedding import FlagModel
from typing import Any

def retrieve_single_query(
    dataset_name: str, 
    query: str, 
    topk: int,
    retrieval_method: str = "bm25",
    embedding_model: FlagModel = None,
    vector: Any = None,
    raw_data: list = None
) -> list:
    
    if retrieval_method == "embedding":
        if embedding_model is None or vector is None or raw_data is None:
            raise ValueError("For embedding retrieval, embedding_model, vector and raw_data must be provided")
        
        query_embedding = embedding_model.encode_queries([query])
            
        _, match_id = vector.search(query_embedding, topk)
        return [raw_data[i]["contents"] if isinstance(raw_data[i], dict) else raw_data[i] for i in match_id[0]]
            
    elif retrieval_method == "bm25":
        from bm25_retriever import retrieve_single_query as bm25_retrieve
        return bm25_retrieve(dataset_name, query, topk)
    else:
        raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

def get_retrieved_docs(
    retrieval_method: str,
    retrieval_docs: list,
) -> str:
    if retrieval_method == "bm25":
        refs = ""
        for i in range(len(retrieval_docs)):
            refs += f"Title: {retrieval_docs[i]['title']}\nText: {retrieval_docs[i]['paragraph_text']} \n\n"
    elif retrieval_method == "embedding":
        refs = "\n".join(retrieval_docs)
    return refs
# Build index for hotpotqa, 2wikimultihopqa, musique, iirc and simpleqa using Embedding model
import sys
import os
import json
import faiss
import pickle

from argparse import ArgumentParser
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from FlagEmbedding import FlagModel

# load environment variables
project_root = os.environ.get("PROJECT_ROOT", "")

sys.path.append(os.path.join(project_root, "src", "build_index"))
from preprocess_datasets import preprocess_hotpotqa, preprocess_2wikimultihopqa, preprocess_musique, preprocess_iirc, preprocess_kilt

def split_text(data, chunk_size, chunk_overlap):
    documents = []
    for record in data:
        if record["title"]:
            combined_text = record["title"] + "\n" + record["text"]
        else:
            combined_text = record["text"]
        documents.append(Document(text=combined_text))

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    contents = [node.text for node in nodes]
    return contents

def build_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding Model Path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wikimultihopqa",
        choices=["2wikimultihopqa", "hotpotqa", "musique", "iirc", "simpleqa"],
        help="Dataset Name",
    )                                        
    parser.add_argument(
        "--input_data", 
        type=str, 
        help="The absolute path of the input data folder"
    )   
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=512, 
        help="chunk size"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=0, 
        help="chunk overlap"
    )
    
    args = parser.parse_args()
    
    # Load Embedding Model
    print("Loading Embedding Model...")
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_devices:
        cuda_devices = None
    else:
        if "," in cuda_devices:
            cuda_devices = cuda_devices.split(",")
            cuda_devices = ["cuda:"+d for d in cuda_devices]
        else:
            cuda_devices = "cuda:" + cuda_devices
    print("Using devices:", cuda_devices)
    model = FlagModel(
        args.model_path, 
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        devices=cuda_devices,
        use_fp16=False
    )
    
    # load corpus corresponding to the dataset
    dataset_name = args.dataset
    print(f"Start loading {dataset_name} docs...")
    docs = []
    if dataset_name == "hotpotqa":
        docs = preprocess_hotpotqa(args.input_data, "embedding")
    elif dataset_name == "2wikimultihopqa":
        docs = preprocess_2wikimultihopqa(args.input_data, "embedding")
    elif dataset_name == "musique":
        docs = preprocess_musique(args.input_data, "embedding")
    elif dataset_name == "iirc":
        docs = preprocess_iirc(args.input_data, "embedding")
    elif dataset_name == "simpleqa":
        docs = preprocess_kilt(args.input_data)
    else:
        raise ValueError(f"Unknown index name: {dataset_name}")
    print(f"Finish loading {len(docs)} docs.")
    
    if dataset_name != "simpleqa":
        docs = split_text(docs, args.chunk_size, args.chunk_overlap)
        with open(os.path.join(args.input_data, f"{dataset_name}_chunk.json"), 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False)
    
    print("Start encode...")
    corpus_embeddings = model.encode_corpus(docs, batch_size=1024)
    
    print("Shape of the corpus embeddings:", corpus_embeddings.shape)
    print("Data type of the embeddings:", corpus_embeddings.dtype)

    print("Start save")
    with open(os.path.join(args.input_data, f"{dataset_name}_{args.model_path.split("/")[-1]}.pkl"), 'ab') as f:
        pickle.dump(corpus_embeddings, f)
    print("Save over")
    
    print("Building index ...")
    build_index(corpus_embeddings, os.path.join(args.input_data, f"{dataset_name}_{args.model_path.split('/')[-1]}.bin"))

if __name__ == "__main__":
    main()
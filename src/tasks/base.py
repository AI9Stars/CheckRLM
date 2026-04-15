import sys
import os
import json
import argparse
import logging
import threading
import time
import faiss
from FlagEmbedding import FlagModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Dict, Any
from openai import OpenAI
from filelock import FileLock

# load environment variables
project_root = os.environ.get("PROJECT_ROOT", "")
vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
reasoning_base_url = os.environ.get("REASONING_BASE_URL", "")

sys.path.append(os.path.join(project_root, "src"))
from evaluate import eval
from utils import setup_logging, load_yaml, load_jsonl, load_txt, load_json, load_tsv, save_json, save_jsonl, safe_api_call
from retrieve import retrieve_single_query, get_retrieved_docs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base with configurable parameters.")
    
    parser.add_argument(
        "--reasoning_model_path",
        type=str,
        required=True,
        help="load reasoning model path."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="hotpotqa",
        choices=["hotpotqa", "2wikimultihopqa", "musique", "simpleqa", "iirc"]
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Dataset file path."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="bge-large-en-v1.5",
        help="embedding model path if the retrieval_method is embedding."
    )
    parser.add_argument(
        "--embedding_model_device",
        type=str,
        default="cuda:7",
        help="embedding model device if the retrieval_method is embedding."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        help="corpus path if the retrieval_method is embedding."
    )
    parser.add_argument(
        "--vector_path",
        type=str,
        help="index path if the retrieval_method is embedding."
    )
    parser.add_argument(
        "--TopK",
        type=int,
        default=3,
        help="Retrieval document TopK."
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        choices=["bm25", "embedding"]
    )
    parser.add_argument(
        "--method",
        type=str,
        default="NoR",
        choices=["NoR", "vanilla"],
        help="Method to answer the question."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
        help="Output root directory."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        help="Resume result file to continue generate and evaluate."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the final answer or not."
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="start debug mode. Only test the first 5 samples in datasets."
    )
    args = parser.parse_args()
    return args
    

def generate_single_step(
    dataset_object: Dict,
    vllm_reasoning_model: OpenAI,
    reasoning_model_path: str,
    method: str, # choices = ["NoR", "vanilla"]
    retrieval_method: str = None,
    **model_kwargs,
) -> str:
    """Reasoning Model Generate Think"""
    
    reasoning_prompt_file = os.path.join(project_root, "src", "prompt", f"{method}.txt")
    reasoning_user_instruction = load_txt(reasoning_prompt_file)
    
    if method == "vanilla":
        refs = get_retrieved_docs(retrieval_method=retrieval_method, retrieval_docs=dataset_object['retrieval_results'])
        messages = [
            {
                "role": "user", 
                "content": reasoning_user_instruction.format(query=dataset_object["question_text"], refs=refs)
            }, 
            {   
                "role": "assistant", 
                "content": "<think>\n"
            }
        ]
    else:
        messages = [
            {
                "role": "user", 
                "content": reasoning_user_instruction.format(query=dataset_object["question_text"])
            }, 
            {   
                "role": "assistant", 
                "content": "<think>\n"
            }
        ]
    
    reasoning_response = safe_api_call(
        vllm_reasoning_model.chat.completions.create,
        model=reasoning_model_path,
        messages=messages,
        temperature=model_kwargs["reasoning_model"]["temperature"],
        top_p=model_kwargs["reasoning_model"]["top_p"],
        max_tokens=model_kwargs["reasoning_model"]["max_tokens"],
        seed=model_kwargs["reasoning_model"]["seed"],
        n=model_kwargs["reasoning_model"]["n"],
        extra_body={"top_k": model_kwargs["reasoning_model"]["top_k"], "repetition_penalty": model_kwargs["reasoning_model"]["repetition_penalty"]}
    )
    
    token_usage = {
        "model": "reasoning_model",
        "prompt_tokens": reasoning_response.usage.prompt_tokens,
        "completion_tokens": reasoning_response.usage.completion_tokens,
        "total_tokens": reasoning_response.usage.total_tokens
    }
    response = reasoning_response.choices[0].message.content
    
    return response, token_usage

def process_single_sample(
    dataset_name: str,
    dataset: Dict,
    vllm_reasoning_model: OpenAI,
    reasoning_model_path: str,
    method: str,
    retrieval_method: str,
    TopK: int,
    model_kwargs: Dict,
    output_queue: Queue,
    vector: Any = None,
    embedding_model: FlagModel = None,
    raw_data: list = None,
) -> None:
    
    # Initialize dataset object
    if dataset_name == "simpleqa":
        dataset_object = {
            "question_id": dataset["question_id"],
            "question_text": dataset["question"],
            "ground_truth": dataset["ground_truth"],
            "retrieval_results": [],
            "think_process": "",
            "predicted_answer": "",
            "token_usage": [],
        }
    else:
        dataset_object = {
            "question_id": dataset["question_id"],
            "question_text": dataset["question_text"],
            "ground_truth": dataset["answers_objects"][0]["spans"],
            "retrieval_results": [],
            "think_process": "",
            "predicted_answer": "",
            "token_usage": [],
        }
    
    # Retrieve documents
    if method == "vanilla":
        dataset_object["retrieval_results"].extend(retrieve_single_query(dataset_name=dataset_name, query=dataset_object["question_text"], topk=TopK, retrieval_method=retrieval_method, embedding_model=embedding_model, vector=vector, raw_data=raw_data))
        
        logging.info(f"{dataset_object['question_id']} Retrieved Docs: {dataset_object['retrieval_results']}")
        
    # generate response with reasoning model
    response, token_usage = generate_single_step(
        dataset_object=dataset_object,
        vllm_reasoning_model=vllm_reasoning_model,
        reasoning_model_path=reasoning_model_path,
        method=method,
        retrieval_method=retrieval_method,
        **model_kwargs,
    )
    
    logging.info(f"{dataset_object['question_id']} Response: {response}")
    
    if "<think>" not in response:
        dataset_object["think_process"] = "<think>\n" + response.split("</think>")[0] + "</think>"
    else:
        dataset_object["think_process"] = response.split("</think>")[0] + "</think>"
    dataset_object["predicted_answer"] = response.split("</think>")[-1]
    dataset_object["token_usage"].append(token_usage)
    
    output_queue.put(dataset_object)
    
    
def run_base(
    reasoning_model_path: str,
    dataset_name: str,
    dataset_file: str,
    retrieval_method: str,
    method: str,
    TopK: int = 3, # retrieval document
    vector: Any = None,
    embedding_model: FlagModel = None,
    raw_data: list = None,
    output_root: str = "outputs",
    resume_path: str = None,
    evaluate: bool = False,
    debug_mode: bool = False,
    **model_kwargs,
) -> str:
    start_time = time.time()
    
    # Load datasets
    datasets = load_jsonl(dataset_file)
    
    if resume_path:
        resume_data = load_jsonl(resume_path)
        processed_ids = {item["question_id"] for item in resume_data}
        datasets = [d for d in datasets if d["question_id"] not in processed_ids]
        saves = resume_data
    else:
        saves = []
    logging.info(f"Loaded {len(datasets)} examples from {dataset_name}")
    
    if debug_mode:
        if len(datasets) > 5:
            datasets = datasets[:5]
        logging.info(f"Running in debug mode.")
        
    # Create output path
    if not resume_path:
        if "/" in reasoning_model_path:
            reasoning_model_name = reasoning_model_path.split("/")[-1]
        else:
            reasoning_model_name = reasoning_model_path
        if method == "vanilla":
            output_dir = os.path.join(project_root, output_root, dataset_name, method, retrieval_method, reasoning_model_name)
        else:
            output_dir = os.path.join(project_root, output_root, dataset_name, method, reasoning_model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "outputs.jsonl")
        logging.info(f"Saving outputs to {output_dir}")
    else:
        output_file_path = resume_path
        output_dir = os.path.dirname(output_file_path)
        logging.info(f"Resume from {output_file_path}...")
    
    # Load reasoning model
    reasoning_model = OpenAI(api_key=vllm_api_key, base_url=reasoning_base_url)
    
    # Create concurrency queue
    output_queue = Queue()
    results = []
    
    # Start save thread
    def save_worker():
        while True:
            item = output_queue.get()
            if item is None: break
            with FileLock(output_file_path + ".lock"):
                with open(output_file_path, "a") as f:
                    f.write(json.dumps(item) + "\n")
            results.append(item)
    
    save_thread = threading.Thread(target=save_worker)
    save_thread.start()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for dataset in datasets:
            future = executor.submit(
                process_single_sample,
                dataset_name=dataset_name,
                dataset=dataset,
                vllm_reasoning_model=reasoning_model,
                reasoning_model_path=reasoning_model_path,
                method=method,
                retrieval_method=retrieval_method,
                TopK=TopK,
                model_kwargs=model_kwargs,
                output_queue=output_queue,
                vector=vector,
                embedding_model=embedding_model,
                raw_data=raw_data,
            )
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Processing failed: {str(e)}")
    
    # Stop save thread
    output_queue.put(None)
    save_thread.join()
    
    final_results = saves + results
    save_jsonl(output_file_path, final_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time:.2f} seconds")
    
    if evaluate:
        eval(output_file_path)

    return output_dir

def main():
    # Parse the command arguments
    args = parse_arguments()
    
    # Load model parameters in yaml file
    reasoning_model_config_file = os.path.join(project_root, "src", "config", "reasoning_model.yaml")
    reasoning_model_config = load_yaml(reasoning_model_config_file)
    
    # Extract arguments
    reasoning_model_path = args.reasoning_model_path
    dataset_name = args.dataset_name
    dataset_file = args.dataset_file
    embedding_model = args.embedding_model
    embedding_model_device = args.embedding_model_device
    corpus_path = args.corpus_path
    vector_path = args.vector_path
    TopK = args.TopK
    retrieval_method = args.retrieval_method
    method = args.method
    output_root = args.output_root
    resume_path = args.resume_path
    if args.evaluate:
        evaluate = True
    else:
        evaluate = False
    if args.debug_mode:
        debug_mode = True
    else:
        debug_mode = False
        
    # Set up logging
    log_file = setup_logging(project_root)
    logging.info(f"Logging to {log_file}")
    
    config = vars(args)
    config.update(reasoning_model_config)
    logging.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in config.items():
        logging.info(f"{key}: {val}")
        
    if retrieval_method == "embedding":
        embedding_model = FlagModel(
            embedding_model,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=False,
            devices=embedding_model_device,
        )
        
        vector = faiss.read_index(vector_path)
            
        if corpus_path.endswith(".json"):
            raw_data = load_json(corpus_path)   
        elif corpus_path.endswith(".tsv"):
            raw_data = load_tsv(corpus_path)
    else:
        vector = None
        embedding_model = None
        raw_data = None
        
    output_dir = run_base(
        reasoning_model_path=reasoning_model_path,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        retrieval_method=retrieval_method,
        method=method,
        TopK=TopK,
        vector=vector,
        embedding_model=embedding_model,
        raw_data=raw_data,
        output_root=output_root,
        resume_path=resume_path,
        evaluate=evaluate,
        debug_mode=debug_mode,
        **reasoning_model_config,
    )
    
    # Merge and save config
    config_file = os.path.join(output_dir, "config.json")
    config["output_dir"] = output_dir
    config["log_file"] = log_file

    if os.path.exists(config_file):
        existing_config = load_json(config_file)
        
        if "log_file" in existing_config:
            if isinstance(existing_config["log_file"], list):
                if log_file not in existing_config["log_file"]:
                    existing_config["log_file"].append(log_file)
            else:
                existing_config["log_file"] = [existing_config["log_file"], log_file]
        else:
            existing_config["log_file"] = [log_file]
                
        save_json(json_file=config_file, json_data=existing_config)
    else:
        config["log_file"] = [log_file]
        save_json(json_file=config_file, json_data=config)
    

if __name__ == "__main__":
    main()
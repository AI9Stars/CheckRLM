import sys
import os
import json
import re
import argparse
import logging
import threading
import time
import faiss
from datetime import timedelta
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
check_base_url = os.environ.get("CHECK_BASE_URL", "")

sys.path.append(os.path.join(project_root, "src"))
from evaluate import eval
from utils import setup_logging, load_yaml, load_jsonl, load_txt, load_json, load_tsv, save_json, save_jsonl, safe_api_call, fix_and_parse_json
from retrieve import retrieve_single_query, get_retrieved_docs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run check_think_offline with configurable parameters.")
    
    parser.add_argument(
        "--reasoning_model_path",
        type=str,
        required=True,
        help="load reasoning model path."
    )
    parser.add_argument(
        "--check_model_path",
        type=str,
        required=True,
        help="load check model path."
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
        "--max_steps",
        type=int,
        default=1,
        help="Maximum steps to check think process."
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
    mode: str, # choices = ["init", "final"]
    **model_kwargs
) -> str:
    """Reasoning Model Generate Think Part"""
    
    reasoning_prompt_file = os.path.join(project_root, "src", "prompt", "NoR.txt")
    reasoning_user_instruction = load_txt(reasoning_prompt_file)
    
    reasoning_response = safe_api_call(
        vllm_reasoning_model.chat.completions.create,
        model=reasoning_model_path,
        messages=[{"role": "user", "content": reasoning_user_instruction.format(query=dataset_object["question_text"])}, {"role": "assistant", "content": "<think>\n" + dataset_object["think_history"]}],
        temperature=model_kwargs["reasoning_model"]["temperature"],
        top_p=model_kwargs["reasoning_model"]["top_p"],
        max_tokens=model_kwargs["reasoning_model"]["max_tokens"],
        seed=model_kwargs["reasoning_model"]["seed"],
        n=model_kwargs["reasoning_model"]["n"],
        stop=["</think>"] if mode == "init" else None,
        extra_body={"top_k": model_kwargs["reasoning_model"]["top_k"], "repetition_penalty": model_kwargs["reasoning_model"]["repetition_penalty"]}
    )
    
    token_usage = {
        "model": "reasoning_model",
        "prompt_tokens": reasoning_response.usage.prompt_tokens,
        "completion_tokens": reasoning_response.usage.completion_tokens,
        "total_tokens": reasoning_response.usage.total_tokens,
        "step_type": mode
    }
    
    reasoning_response = reasoning_response.choices[0].message.content
    
    return reasoning_response, token_usage

def check_single_step(
    think: str,
    dataset_name: str,
    dataset_object: Dict,
    retrieval_method: str,
    TopK: int,
    vllm_check_model: OpenAI,
    check_model_path: str,
    vector: Any = None,
    embedding_model: FlagModel = None,
    raw_data: list = None,
    **model_kwargs
) -> Dict:
    """Check Model check and refine Think Part"""
    
    check_prompt_file = os.path.join(project_root, "src", "prompt", "check.txt")
    system_prompt = load_txt(check_prompt_file)
    user_prompt = f"Question: {dataset_object['question_text']}\nReasoning Process: {think}\n\nFactual Claim List:"
    
    messages = [
        {
            "role": "system", 
            "content": system_prompt
        }, 
        {
            "role": "user", 
            "content": user_prompt
        }
    ]
    
    check_response = safe_api_call(
        vllm_check_model.chat.completions.create,
        model=check_model_path,
        messages=messages,
        temperature=model_kwargs["check_model"]["temperature"],
        top_p=model_kwargs["check_model"]["top_p"],
        max_tokens=model_kwargs["check_model"]["max_tokens"],
        seed=model_kwargs["check_model"]["seed"],
        n=model_kwargs["check_model"]["n"],
        extra_body={"top_k": model_kwargs["check_model"]["top_k"], "repetition_penalty": model_kwargs["check_model"]["repetition_penalty"], "chat_template_kwargs": {"enable_thinking": False}}
    )
    
    token_usage = {
        "model": "check_model",
        "prompt_tokens": check_response.usage.prompt_tokens,
        "completion_tokens": check_response.usage.completion_tokens,
        "total_tokens": check_response.usage.total_tokens,
        "step_type": "check"
    }
    dataset_object["token_usage"].append(token_usage)
    
    check_response = check_response.choices[0].message.content
    
    if "Factual Claim List:" in check_response:
        check_response = check_response.split("Factual Claim List:")[-1]
    
    logging.info(f"{dataset_object['question_id']} Check Response: {check_response}")
        
    pattern = r'\[[^]]*\]'
    matches = re.findall(pattern, check_response)
    subqueries = []
    for match in matches:
        json_list = fix_and_parse_json(match)
        if json_list == [] or json_list == [""]:
            continue
        else:
            subqueries.extend(json_list)
    
    logging.info(f"{dataset_object['question_id']} Subqueries: {subqueries}")
    
    cache_search_results = []
    for subquery in subqueries:
        if subquery in dataset_object["search_queries"]:
            cache_search_results.extend(dataset_object["search_results"][dataset_object["search_queries"][dataset_object["search_queries"].index(subquery)]])
        else:
            search_results = retrieve_single_query(
                dataset_name=dataset_name, 
                query=subquery, 
                topk=TopK,
                retrieval_method=retrieval_method,
                embedding_model=embedding_model,
                vector=vector,
                raw_data=raw_data
            )
            cache_search_results.extend(search_results)
            dataset_object["search_queries"].append(subquery)
            dataset_object["search_results"][subquery] =  search_results
    cache_search_results.extend(dataset_object["search_results"][dataset_object["question_text"]])
    if retrieval_method == "bm25":
        cache_search_results = [dict(t) for t in {tuple(c.items()) for c in cache_search_results}]
    else:
        cache_search_results = list(set(cache_search_results))
    
    refs = get_retrieved_docs(retrieval_method=retrieval_method, retrieval_docs=cache_search_results)
    
    refine_prompt_file = os.path.join(project_root, "src", "prompt", "refine.txt")
    system_prompt = load_txt(refine_prompt_file)
    user_prompt = f"Retrieved documents: {refs}\nReasoning Process: {think}\n\nProvide your modified reasoning process:"
    
    messages = [
        {
            "role": "system", 
            "content": system_prompt
        }, 
        {
            "role": "user", 
            "content": user_prompt
        }
    ]
    
    refine_response = safe_api_call(
        vllm_check_model.chat.completions.create,
        model=check_model_path,
        messages=messages,
        temperature=model_kwargs["check_model"]["temperature"],
        top_p=model_kwargs["check_model"]["top_p"],
        max_tokens=model_kwargs["check_model"]["max_tokens"],
        seed=model_kwargs["check_model"]["seed"],
        n=model_kwargs["check_model"]["n"],
        extra_body={"top_k": model_kwargs["check_model"]["top_k"], "repetition_penalty": model_kwargs["check_model"]["repetition_penalty"], "chat_template_kwargs": {"enable_thinking": False}}
    )
    
    token_usage = {
        "model": "check_model",
        "prompt_tokens": refine_response.usage.prompt_tokens,
        "completion_tokens": refine_response.usage.completion_tokens,
        "total_tokens": refine_response.usage.total_tokens,
        "step_type": "refine"
    }
    dataset_object["token_usage"].append(token_usage)
    
    refine_response = refine_response.choices[0].message.content
    
    if "Reasoning Process:" in refine_response:
        refine_response = refine_response.split("Reasoning Process:")[-1]
    if "Provide your modified reasoning process:" in refine_response:
        refine_response = refine_response.split("Provide your modified reasoning process:")[-1]
        
    logging.info(f"{dataset_object['question_id']} Refine Response: {refine_response}")
    
    dataset_object["think_history"] = refine_response
    
    return dataset_object


def process_single_sample(
    dataset_name: str,
    dataset: Dict,
    vllm_reasoning_model: OpenAI,
    vllm_check_model: OpenAI,
    reasoning_model_path: str,
    check_model_path: str,
    retrieval_method: str,
    TopK: int,
    model_kwargs: Dict,
    output_queue: Queue,
    max_steps: int = 1,
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
            "think_history": "",
            "search_queries": [dataset["question"]],
            "search_results": {dataset["question"]: retrieve_single_query(dataset_name=dataset_name, query=dataset["question"], topk=TopK, retrieval_method=retrieval_method, embedding_model=embedding_model, vector=vector, raw_data=raw_data)},
            "predicted_answer": "",
            "steps": 0,
            "token_usage": [],
        }
    else:
        dataset_object = {
            "question_id": dataset["question_id"],
            "question_text": dataset["question_text"],
            "ground_truth": dataset["answers_objects"][0]["spans"],
            "think_history": "",
            "search_queries": [dataset["question_text"]],
            "search_results": {dataset["question_text"]: retrieve_single_query(dataset_name=dataset_name, query=dataset["question_text"], topk=TopK, retrieval_method=retrieval_method, embedding_model=embedding_model, vector=vector, raw_data=raw_data)},
            "predicted_answer": "",
            "steps": 0,
            "token_usage": [],
        }
    
    # Initialize first think part
    think_part, token_usage = generate_single_step(
        dataset_object=dataset_object,
        vllm_reasoning_model=vllm_reasoning_model,
        reasoning_model_path=reasoning_model_path,
        mode="init",
        **model_kwargs
    )
    logging.info(f"{dataset_object['question_id']} Init Think Part: {think_part}")
    dataset_object["token_usage"].append(token_usage)
    
    for _ in range(max_steps): 
        dataset_object = check_single_step(
            query=dataset_object["question_text"],
            think=think_part.replace("<think>", "").replace("</think>", ""),
            dataset_name=dataset_name,
            dataset_object=dataset_object,
            retrieval_method=retrieval_method,
            TopK=TopK,
            vllm_check_model=vllm_check_model,
            check_model_path=check_model_path,
            vector=vector,
            embedding_model=embedding_model,
            raw_data=raw_data,
            **model_kwargs
        )
        think_part = dataset_object["think_history"]
    
    dataset_object["steps"] = max_steps
    dataset_object["think_history"] = "<think>\n" + dataset_object["think_history"] + "\n</think>"
        
    final_answer, token_usage = generate_single_step(
        dataset_object=dataset_object,
        vllm_reasoning_model=vllm_reasoning_model,
        reasoning_model_path=reasoning_model_path,
        mode="final",
        **model_kwargs
    )
    dataset_object["predicted_answer"] = final_answer
    dataset_object["token_usage"].append(token_usage)
    output_queue.put(dataset_object)
    
    
def check_think_offline(
    reasoning_model_path: str,
    check_model_path: str,
    dataset_name: str,
    dataset_file: str,
    retrieval_method: str,
    TopK: int = 3, # retrieval document
    vector: Any = None,
    embedding_model: FlagModel = None,
    raw_data: list = None,
    output_root: str = "outputs",
    resume_path: str = None,
    evaluate: bool = False,
    debug_mode: bool = False,
    max_steps: int = 10,
    **model_kwargs,
) -> str:
    # Start time tracking
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
        if "/" in check_model_path:
            check_model_name = check_model_path.split("/")[-1]
        else:
            check_model_name = check_model_path
        output_dir = os.path.join(project_root, output_root, dataset_name, 'refine_think', f"offline_{max_steps}", f"{reasoning_model_name}-{check_model_name}", retrieval_method)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "outputs.jsonl")
        logging.info(f"Saving outputs to {output_dir}")
    else:
        output_file_path = resume_path
        output_dir = os.path.dirname(output_file_path)
        logging.info(f"Resume from {output_file_path}...")
    
    # Load reasoning model, check model and tokenizer
    reasoning_model = OpenAI(api_key=vllm_api_key, base_url=reasoning_base_url)
    check_model = OpenAI(api_key=vllm_api_key, base_url=check_base_url)
    
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
                vllm_check_model=check_model,
                reasoning_model_path=reasoning_model_path,
                check_model_path=check_model_path,
                retrieval_method=retrieval_method,
                TopK=TopK,
                model_kwargs=model_kwargs,
                output_queue=output_queue,
                max_steps=max_steps,
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
    
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {timedelta(seconds=total_time)}")
    
    if evaluate:
        eval(output_file_path)

    return output_dir

def main():
    # Parse the command arguments
    args = parse_arguments()
    
    # Load model parameters in yaml file
    reasoning_model_config_file = os.path.join(project_root, "src", "config", "reasoning_model.yaml")
    check_model_config_file = os.path.join(project_root, "src", "config", "check_model.yaml")
    reasoning_model_config = load_yaml(reasoning_model_config_file)
    check_model_config = load_yaml(check_model_config_file)
    
    # Extract arguments
    reasoning_model_path = args.reasoning_model_path
    check_model_path = args.check_model_path
    dataset_name = args.dataset_name
    dataset_file = args.dataset_file
    embedding_model = args.embedding_model
    embedding_model_device = args.embedding_model_device
    corpus_path = args.corpus_path
    vector_path = args.vector_path
    TopK = args.TopK
    retrieval_method = args.retrieval_method
    max_steps = args.max_steps
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
    config.update(check_model_config)
    logging.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in config.items():
        logging.info(f"{key}: {val}")
        
    if retrieval_method == "embedding":
        embedding_model = FlagModel(
            embedding_model, query_instruction_for_retrieval="Represent this sentence for searching relevant passages:", use_fp16=False, 
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
        
    output_dir = check_think_offline(
        reasoning_model_path=reasoning_model_path,
        check_model_path=check_model_path,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        retrieval_method=retrieval_method,
        TopK=TopK,
        vector=vector,
        embedding_model=embedding_model,
        raw_data=raw_data,
        output_root=output_root,
        resume_path=resume_path,
        evaluate=evaluate,
        debug_mode=debug_mode,
        max_steps=max_steps,
        **{**reasoning_model_config, **check_model_config},
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
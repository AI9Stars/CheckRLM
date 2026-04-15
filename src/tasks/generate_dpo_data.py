import sys
import os
import json
import re
import random
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Dict, List
from openai import OpenAI
from filelock import FileLock

# load environment variables
project_root = os.environ.get("PROJECT_ROOT", "")
vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
reasoning_base_url = os.environ.get("REASONING_BASE_URL", "")
check_base_url = os.environ.get("CHECK_BASE_URL", "")
gpt_api_key = os.environ.get("OPENAI_API_KEY", "")
gpt_base_url = os.environ.get("OPENAI_BASE_URL", "")

sys.path.append(os.path.join(project_root, "src"))
from bm25_retriever import retrieve_single_query
from utils import setup_logging, load_yaml, load_jsonl, load_json, load_txt, save_json, save_jsonl, safe_api_call, fix_and_parse_json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate DPO data with configurable parameters.")
    
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
        "--evaluate_model",
        type=str,
        required=True,
        default="gpt-4o-mini",
        help="Evaluate model to score the samples."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="2wikimultihopqa",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Dataset file path."
    )
    parser.add_argument(
        "--dpo_number",
        type=int,
        default=10,
        help="The number of samples in DPO dataset."
    )
    parser.add_argument(
        "--TopK",
        type=int,
        default=3,
        help="Retrieval document TopK."
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum steps to check think process."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="DPO_Training_Data",
        help="Output root directory."
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        help="Resume result file to continue generate and evaluate."
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
        stop=None if mode == "final" else ["\n\n"],
        extra_body={"top_k": model_kwargs["reasoning_model"]["top_k"], "repetition_penalty": model_kwargs["reasoning_model"]["repetition_penalty"]}
    )
    
    return reasoning_response.choices[0].message.content

def check_single_step(
    think: str,
    dataset_object: Dict,
    vllm_check_model: OpenAI,
    check_model_path: str,
    evaluate_model: OpenAI,
    evaluate_model_name: str,
    **model_kwargs
) -> Dict:
    """Sampling the Check Model check responses and score the best and worst responses."""
    
    check_prompt_file = os.path.join(project_root, "src", "prompt", "check.txt")
    system_prompt = load_txt(check_prompt_file)
    user_prompt = f"Question: {dataset_object['question_text']}\nReasoning Process: {think}\n\nFactual Claim List:"
    
    # Get parameter lists
    temperatures = model_kwargs["check_model"]["temperature"] 
    top_ps = model_kwargs["check_model"]["top_p"]
    
    # Prepare all combinations
    param_combinations = [
        {"temperature": t, "top_p": p} 
        for t in temperatures 
        for p in top_ps
    ]
    
    candidate_check_responses = []
    gen_responses = []
    for i, param in enumerate(param_combinations):
        check_response = safe_api_call(
            vllm_check_model.chat.completions.create,
            model=check_model_path,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=param["temperature"],
            top_p=param["top_p"],
            max_tokens=model_kwargs["check_model"]["max_tokens"],
            seed=model_kwargs["check_model"]["seed"],
            n=model_kwargs["check_model"]["n"],
            extra_body={"top_k": model_kwargs["check_model"]["top_k"], "repetition_penalty": model_kwargs["check_model"]["repetition_penalty"], "chat_template_kwargs": {"enable_thinking": False}}
        )
        check_response = check_response.choices[0].message.content
        
        if "Factual Claim List:" in check_response:
            check_response = check_response.split("Factual Claim List:")[-1]
        
        candidate_check_responses.append(check_response)
        gen_responses.append({"id": i, "response": check_response, "temperature": param["temperature"], "top_p": param["top_p"]})
    
    candidate_check_responses = list(set(candidate_check_responses))
    candidate_check_responses = [{"id": i, "content": candidate_check_response} for i, candidate_check_response in enumerate(candidate_check_responses)]
    logging.info(f"{dataset_object['question_id']} Check Responses: {gen_responses}")
    
    if len(candidate_check_responses) == 1:
        return {"chosen": candidate_check_responses[0]["content"]}
    else:
        # return a dict like {"best_response": "", "worst_response": ""}
        check_response_pairs = score_check_response(
            query_id=dataset_object["question_id"],
            query=dataset_object["question_text"],
            think=think,
            check_responses=candidate_check_responses,
            evaluate_model=evaluate_model,
            evaluate_model_name=evaluate_model_name
        )
        
        if "best_response" and "worst_response" in check_response_pairs:
            dpo_sample = {
                "id": dataset_object["question_id"],
                "question": dataset_object["question_text"],
                "prompt": system_prompt + "\n\n" + user_prompt,
                "chosen": check_response_pairs["best_response"],
                "rejected": check_response_pairs["worst_response"],
                "data_type": "check",
                "gen_response_list": gen_responses,
            }
            return dpo_sample
        elif "response" in check_response_pairs:
            return {"chosen": check_response_pairs["response"]}
        else:
            return {}

def score_check_response(
    query_id: str,
    query: str,
    think: str,
    check_responses: List[Dict],
    evaluate_model: OpenAI,
    evaluate_model_name: str,
    max_retries: int = 3,
) -> Dict:
    """Score the check responses, filter the best and worst one."""
    
    evaluate_prompt_file = os.path.join(project_root, "src", "prompt", "evaluate_check.txt")
    evaluate_user_instruction = load_txt(evaluate_prompt_file)
    
    check_pairs = {}
    for _ in range(max_retries):
        evaluate_response = safe_api_call(
            evaluate_model.chat.completions.create,
            model=evaluate_model_name,
            messages=[{"role": "user", "content": evaluate_user_instruction.format(query=query, think=think, check_responses=check_responses)}],
            temperature=0,
            top_p=0.95,
            response_format={"type": "json_object"}
        )
        evaluate_response = evaluate_response.choices[0].message.content
        # Response format: {"best_id": , "worst_id": }
        logging.info(f"{query_id}: Check Evaluate: {evaluate_response}")
        evaluate_response = evaluate_response.replace("```json\n", "").replace("\n```", "") 
        try:
            evaluate_response = json.loads(evaluate_response)
            if evaluate_response == {}:
                break
            else:
                for check_response in check_responses:
                    if evaluate_response["best_id"] == evaluate_response["worst_id"] and check_response["id"] == evaluate_response["best_id"]:
                        check_pairs["response"] = check_response["content"]
                    else:
                        if check_response["id"] == evaluate_response["best_id"]:
                            check_pairs["best_response"] = check_response["content"]
                        if check_response["id"] == evaluate_response["worst_id"]:
                            check_pairs["worst_response"] = check_response["content"]
                break
        except json.JSONDecodeError:
            logging.error(f"{evaluate_response} JSON Decode Fail!")
    
    return check_pairs

def search_queries(
    dataset_name: str,
    dataset_object: Dict,
    check_response: str,
    TopK: int = 3,
):
    pattern = r'\[[^]]*\]'
    matches = re.findall(pattern, str(check_response))
    subqueries = []
    for match in matches:
        json_list = fix_and_parse_json(match)
        if json_list == [] or json_list == [""]:
            subqueries = []
            break
        else:
            subqueries.extend(json_list)
    
    if subqueries == []:
        dataset_object["cache_docs"] = []
        return dataset_object
    
    cache_search_results = []
    for subquery in subqueries:
        if subquery in dataset_object["search_queries"]:
            cache_search_results.extend(dataset_object["search_results"][dataset_object["search_queries"][dataset_object["search_queries"].index(subquery)]])
        else:
            search_results = retrieve_single_query(
                index_name=dataset_name, 
                query=subquery, 
                topk=TopK
            )
            cache_search_results.extend(search_results)
            dataset_object["search_queries"].append(subquery)
            dataset_object["search_results"][subquery] =  search_results
    cache_search_results.extend(dataset_object["search_results"][dataset_object["question_text"]])
    cache_search_results = [dict(t) for t in {tuple(c.items()) for c in cache_search_results}]
    dataset_object["cache_docs"] = cache_search_results
    return dataset_object

def refine_single_step(
    think: str,
    dataset_object: Dict,
    vllm_check_model: OpenAI,
    check_model_path: str,
    evaluate_model: OpenAI,
    evaluate_model_name: str,
    **model_kwargs
) -> Dict:
    """Sampling the Refine Model refine responses and score the best and worst responses."""
    
    refine_prompt_file = os.path.join(project_root, "src", "prompt", "refine.txt")
    system_prompt = load_txt(refine_prompt_file)
    
    refs = ""
    for i in range(len(dataset_object["cache_docs"])):
        refs += f"Passage #{i+1} Title: {dataset_object['cache_docs'][i]['title']}\nPassage #{i+1} Text: {dataset_object['cache_docs'][i]['paragraph_text']} \n\n"
    user_prompt = f"Retrieved documents: {refs}\nReasoning Process: {think}\n\nProvide your modified reasoning process:"
    
    # Get parameter lists
    temperatures = model_kwargs["check_model"]["temperature"] 
    top_ps = model_kwargs["check_model"]["top_p"]
    
    # Prepare all combinations
    param_combinations = [
        {"temperature": t, "top_p": p} 
        for t in temperatures 
        for p in top_ps
    ]
    
    candidate_refine_responses = []
    gen_responses = []
    for i, param in enumerate(param_combinations):
        refine_response = safe_api_call(
            vllm_check_model.chat.completions.create,
            model=check_model_path,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=param["temperature"],
            top_p=param["top_p"],
            max_tokens=model_kwargs["check_model"]["max_tokens"],
            seed=model_kwargs["check_model"]["seed"],
            n=model_kwargs["check_model"]["n"],
            extra_body={"top_k": model_kwargs["check_model"]["top_k"], "repetition_penalty": model_kwargs["check_model"]["repetition_penalty"], "chat_template_kwargs": {"enable_thinking": False}}
        )
        refine_response = refine_response.choices[0].message.content
        
        if "Reasoning Process:" in refine_response:
            refine_response = refine_response.split("Reasoning Process:")[-1]
        if "Provide your modified reasoning process:" in refine_response:
            refine_response = refine_response.split("Provide your modified reasoning process:")[-1]
        
        candidate_refine_responses.append(refine_response)
        gen_responses.append({"id": i, "response": refine_response, "temperature": param["temperature"], "top_p": param["top_p"]})
        
    candidate_refine_responses = list(set(candidate_refine_responses))
    candidate_refine_responses = [{"id": i, "content": candidate_refine_response} for i, candidate_refine_response in enumerate(candidate_refine_responses)]
    logging.info(f"{dataset_object['question_id']}, Refine Response: {candidate_refine_responses}")
    
    if len(candidate_refine_responses) == 1:
        return {"chosen": candidate_refine_responses[0]["content"]}
    else:
        # return a dict like {"best_response": "", "worst_response": ""}
        refine_response_pairs = score_refine_response(
            query_id=dataset_object["question_id"],
            refs=refs,
            think=think,
            refine_responses=candidate_refine_responses,
            evaluate_model=evaluate_model,
            evaluate_model_name=evaluate_model_name
        )
        logging.info(f"{dataset_object['question_id']}, Evaluate Refine Response: {refine_response_pairs}")
        
        if "best_response" and "worst_response" in refine_response_pairs:
            dpo_sample = {
                "id": dataset_object["question_id"],
                "question": dataset_object["question_text"],
                "prompt": system_prompt + "\n\n" + user_prompt,
                "chosen": refine_response_pairs["best_response"],
                "rejected": refine_response_pairs["worst_response"],
                "data_type": "refine",
                "gen_response_list": gen_responses,
            }
            return dpo_sample
        elif "response" in refine_response_pairs:
            return {"chosen": refine_response_pairs["response"]}
        else:
            return {}
    
def score_refine_response(
    query_id: str,
    refs: str,
    think: str,
    refine_responses: List[Dict],
    evaluate_model: OpenAI,
    evaluate_model_name: str,
    max_retries: int = 3,
) -> Dict:
    """Score the refine responses, filter the best and worst one."""
    
    evaluate_prompt_file = os.path.join(project_root, "src", "prompt", "evaluate_refine.txt")
    evaluate_user_instruction = load_txt(evaluate_prompt_file)
    
    refine_pairs = {}
    for _ in range(max_retries):
        evaluate_response = safe_api_call(
            evaluate_model.chat.completions.create,
            model=evaluate_model_name,
            messages=[{"role": "user", "content": evaluate_user_instruction.format(refs=refs, think=think, refine_responses=refine_responses)}],
            temperature=0,
            top_p=0.95,
            response_format={"type": "json_object"}
        )
        evaluate_response = evaluate_response.choices[0].message.content
        # Response format: {"best_id": , "worst_id": }
        logging.info(f"{query_id}: Refine Evaluate: {evaluate_response}")
        evaluate_response = evaluate_response.replace("```json\n", "").replace("\n```", "") 
        try:
            evaluate_response = json.loads(evaluate_response)
            if evaluate_response == {}:
                break
            else:
                for refine_response in refine_responses:
                    if evaluate_response["best_id"] == evaluate_response["worst_id"] and refine_response["id"] == evaluate_response["best_id"]:
                        refine_pairs["response"] = refine_response["content"]
                    else:
                        if refine_response["id"] == evaluate_response["best_id"]:
                            refine_pairs["best_response"] = refine_response["content"]
                        if refine_response["id"] == evaluate_response["worst_id"]:
                            refine_pairs["worst_response"] = refine_response["content"]
                break
        except json.JSONDecodeError:
            logging.error(f"{evaluate_response} JSON Decode Fail!")
    
    return refine_pairs

def process_single_sample(
    dataset_name: str,
    dataset: Dict,
    vllm_reasoning_model: OpenAI,
    vllm_check_model: OpenAI,
    evaluate_model: OpenAI,
    reasoning_model_path: str,
    check_model_path: str,
    evaluate_model_name: str,
    TopK: int,
    model_kwargs: Dict,
    output_queue: Queue,
    max_steps: int = 10,
) -> None:
    
    # Initialize dataset object 
    dataset_object = {
        "question_id": dataset["_id"],
        "question_text": dataset["question"],
        "think_history": "",
        "cache_docs": [],
        "search_queries": [dataset["question"]],
        "search_results": {dataset["question"]: retrieve_single_query(index_name=dataset_name, query=dataset["question"], topk=TopK)},
        "steps": 0,
    }
    
    dpo_check = []
    dpo_refine = []
    check_fail_flag = False
    refine_fail_flag = False
    
    current_step = 0
    # Initialize first think part
    think_part = generate_single_step(
        dataset_object=dataset_object,
        vllm_reasoning_model=vllm_reasoning_model,
        reasoning_model_path=reasoning_model_path,
        mode="init",
        **model_kwargs
    )
    dataset_object["think_history"] = think_part
    logging.info(f"{dataset_object['question_id']} Init Think Part: {think_part}")
    
    while current_step < max_steps:
        current_step += 1
        if "</think>" in think_part:
            dpo_check_sample = check_single_step(
                think=think_part.split("</think>")[0],
                dataset_object=dataset_object,
                vllm_check_model=vllm_check_model,
                check_model_path=check_model_path,
                evaluate_model=evaluate_model,
                evaluate_model_name=evaluate_model_name,
                **model_kwargs
            )
            if dpo_check_sample != {}:
                if "id" in dpo_check_sample:
                    dpo_check_sample.update({"step": current_step})
                    dpo_check.append(dpo_check_sample)
                dataset_object = search_queries(
                    dataset_name=dataset_name, 
                    dataset_object=dataset_object, 
                    check_response=dpo_check_sample["chosen"],
                    TopK=TopK
                )
                if dataset_object["cache_docs"] == []:
                    dataset_object["think_history"] += think_part.rstrip("\n") + "\n\n"
                else:
                    dpo_refine_sample = refine_single_step(
                        think=think_part.split("</think>")[0],
                        dataset_object=dataset_object,
                        vllm_check_model=vllm_check_model,
                        check_model_path=check_model_path,
                        evaluate_model=evaluate_model,
                        evaluate_model_name=evaluate_model_name,
                        **model_kwargs
                    )
                    if dpo_refine_sample != {}:
                        if "id" in dpo_refine_sample:
                            dpo_refine_sample.update({"step": current_step})
                            dpo_refine.append(dpo_refine_sample)
                        dataset_object["think_history"] += dpo_refine_sample["chosen"].rstrip("\n") + "\n\n"
                    else:
                        refine_fail_flag = True
                        break
            else:
                check_fail_flag = True
                break
            logging.info(f"{current_step} turns: {dataset_object['question_id']}")
            break
        
        dpo_check_sample = check_single_step(
            think=think_part.replace("<think>", ""),
            dataset_object=dataset_object,
            vllm_check_model=vllm_check_model,
            check_model_path=check_model_path,
            evaluate_model=evaluate_model,
            evaluate_model_name=evaluate_model_name,
            **model_kwargs
        )
        if dpo_check_sample != {}:
            if "id" in dpo_check_sample:
                dpo_check_sample.update({"step": current_step})
                dpo_check.append(dpo_check_sample)
            dataset_object = search_queries(
                dataset_name=dataset_name, 
                dataset_object=dataset_object, 
                check_response=dpo_check_sample["chosen"],
                TopK=TopK
            )
            if dataset_object["cache_docs"] == []:
                dataset_object["think_history"] += think_part.rstrip("\n") + "\n\n"
            else:
                dpo_refine_sample = refine_single_step(
                    think=think_part.split("</think>")[0],
                    dataset_object=dataset_object,
                    vllm_check_model=vllm_check_model,
                    check_model_path=check_model_path,
                    evaluate_model=evaluate_model,
                    evaluate_model_name=evaluate_model_name,
                    **model_kwargs
                )
                if dpo_refine_sample != {}:
                    if "id" in dpo_refine_sample:
                        dpo_refine_sample.update({"step": current_step})
                        dpo_refine.append(dpo_refine_sample)
                    dataset_object["think_history"] += dpo_refine_sample["chosen"].rstrip("\n") + "\n\n"
                else:
                    refine_fail_flag = True
                    break
        else:
            check_fail_flag = True
            break
        
        think_part = generate_single_step(
            dataset_object=dataset_object,
            vllm_reasoning_model=vllm_reasoning_model,
            reasoning_model_path=reasoning_model_path,
            mode="init",
            **model_kwargs
        )
    
        logging.info(f"{dataset_object['question_id']} {current_step} step Think Part: {think_part}")
        if not think_part.replace("</think>", "").replace("\n", ""):
            logging.warning(f"{dataset_object['question_id']} think part is None!")
            break
    
    if check_fail_flag:
        logging.error(f"{dataset_object['question_id']} check fail!")
    if refine_fail_flag:
        logging.error(f"{dataset_object['question_id']} refine fail!")
    
    dpo_data = dpo_check + dpo_refine
    output_queue.put(dpo_data)
    
    
def Generate_DPO_Training_Data(
    reasoning_model_path: str,
    check_model_path: str,
    evaluate_model_name: str,
    dataset_name: str,
    dataset_file: str,
    TopK: int = 3, # retrieval document
    output_root: str = "outputs",
    resume_path: str = None,
    dpo_number: int = 10,
    debug_mode: bool = False,
    max_steps: int = 10,
    **model_kwargs,
) -> str:
    # Load datasets
    assert dataset_file.endswith(".json"), "Dataset File must be json files!"
    datasets = load_json(dataset_file)
    
    if resume_path:
        resume_data = load_jsonl(resume_path)
        processed_ids = {item["_id"] for item in resume_data}
        datasets = [d for d in datasets if d["_id"] not in processed_ids]
        saves = resume_data
    else:
        saves = []
    logging.info(f"Loaded {len(datasets)} examples from {dataset_name}")
    
    if debug_mode:
        if len(datasets) > 5:
            datasets = datasets[:5]
        logging.info(f"Running in debug mode.")
    else:
        # Random select samples in DPO training dataset
        random.seed(42)
        datasets = random.sample(datasets, dpo_number)
        logging.info(f"random select {len(datasets)} in {dataset_name}.")
        
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
        output_dir = os.path.join(project_root, output_root, dataset_name, f"{reasoning_model_name}-{check_model_name}")
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "dpo_training_data.jsonl")
        logging.info(f"Saving outputs to {output_dir}")
    else:
        output_file_path = resume_path
        output_dir = os.path.dirname(output_file_path)
        logging.info(f"Resume from {output_file_path}...")
    
    # Load reasoning model, check model and evaluate model
    reasoning_model = OpenAI(api_key=vllm_api_key, base_url=reasoning_base_url)
    check_model = OpenAI(api_key=vllm_api_key, base_url=check_base_url)
    evaluate_model = OpenAI(api_key=gpt_api_key, base_url=gpt_base_url)
    
    # Create concurrency queue
    output_queue = Queue()
    results = []
    
    # Start save thread
    def save_worker():
        while True:
            items = output_queue.get()
            if items is None: break
            with FileLock(output_file_path + ".lock"):
                with open(output_file_path, "a") as f:
                    for item in items:
                        f.write(json.dumps(item) + "\n")
            results.extend(items)
    
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
                evaluate_model=evaluate_model,
                reasoning_model_path=reasoning_model_path,
                check_model_path=check_model_path,
                evaluate_model_name=evaluate_model_name,
                TopK=TopK,
                model_kwargs=model_kwargs,
                output_queue=output_queue,
                max_steps=max_steps,
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
    evaluate_model_name = args.evaluate_model
    dataset_name = args.dataset_name
    dataset_file = args.dataset_file
    dpo_number = args.dpo_number
    TopK = args.TopK
    max_steps = args.max_steps
    output_root = args.output_root
    resume_path = args.resume_path
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
        
    output_dir = Generate_DPO_Training_Data(
        reasoning_model_path=reasoning_model_path,
        check_model_path=check_model_path,
        evaluate_model_name=evaluate_model_name,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        TopK=TopK,
        output_root=output_root,
        resume_path=resume_path,
        dpo_number=dpo_number,
        debug_mode=debug_mode,
        max_steps=max_steps,
        **{**reasoning_model_config, **check_model_config},
    )
    
    # Merge and save config
    config_file = os.path.join(output_dir, "config.json")
    config["output_dir"] = output_dir
    config["log_file"] = log_file
    save_json(json_file=config_file, json_data=config)
    

if __name__ == "__main__":
    main()
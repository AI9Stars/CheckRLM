import os
import json
import yaml
import logging
import datetime
import pandas as pd
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_api_call(api_func, *args, **kwargs):
    return api_func(*args, **kwargs)

def load_jsonl(jsonl_file: str):
    """load jsonl file if file format is jsonline."""
    
    with open(jsonl_file, "r") as file:
        return [json.loads(line) for line in file]
    
def load_yaml(yaml_file: str) -> Dict[str, Any]:
    """Load yaml json"""
    
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)
    
def load_txt(txt_file: str) -> str:
    """Load txt file"""
    
    with open(txt_file, "r") as f:
        return f.read()

def load_json(json_file: str) -> List[Dict]:
    """load json file"""
    
    with open(json_file, "r") as f:
        return json.load(f)
    
def load_csv(csv_file: str):
    """load csv file"""
    
    datas = pd.read_csv(csv_file)
    datas.dropna(axis=1, how='all', inplace=True)
    
    return datas
    
def load_tsv(tsv_file: str):
    with open(tsv_file, "r", encoding="utf-8") as file:
        corpus = file.readlines()
        corpus = [line.strip("\n") for line in corpus]
    return corpus

def save_json(json_file: str, json_data: Dict) -> None:
    """Save json to json file."""
    
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)
        
def save_jsonl(jsonl_file: str, data: List[Dict]) -> None:
    """Save data to jsonl file."""
    
    with open(jsonl_file, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")
        
def setup_logging(log_root: str) -> str:
    """Set up logging file and return logging file path."""
    
    log_dir = os.path.join(log_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)]
    )
    return log_file

def fix_and_parse_json(broken_str):
    try:
        return json.loads(broken_str)
    except json.JSONDecodeError:
        pass

    content = broken_str.strip(' []')
    
    elements = []
    current = []
    in_quotes = False
    escape = False
    for char in content:
        if char == '\\' and not escape:
            escape = True
            current.append(char)
            continue
        if char == '"' and not escape:
            in_quotes = not in_quotes
        if char == ',' and not in_quotes and not escape:
            elements.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
        escape = False  
    
    if current:
        elements.append(''.join(current).strip())

    fixed_elements = []
    for elem in elements:
        elem = elem.strip()
        
        if elem.startswith('"'):
            elem = elem[1:]
        if elem.endswith('"'):
            elem = elem[:-1]
        
        try:
            json.loads(f'"{elem}"')
            fixed_elem = f'"{elem}"'
        except json.JSONDecodeError:
            fixed_elem = json.dumps(elem)
        
        fixed_elements.append(fixed_elem)

    try:
        fixed_json = f'[{",".join(fixed_elements)}]'
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}, original content: {broken_str}, fixed content: {fixed_json}")
        return []
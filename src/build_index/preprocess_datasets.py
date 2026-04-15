import os
import bz2
import io
import json
import html
import dill
import base58
import hashlib
import random

from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from itertools import chain
from typing import Any
from functools import partial
from bs4 import BeautifulSoup


def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()

def process_line(line, index_name):
    if index_name == "hotpotqa":
        data = json.loads(line)
    else:
        data = line
    item = {
        "id": data["id"],
        "url": data["url"] if "url" in data else "",    
        "title": data["title"],
        "title_unescape": html.unescape(data["title"]),
        "text": "".join(data["text"]) if isinstance(data["text"], list) else data["text"],
        "title_bigram": html.unescape(data["title"]),
        "title_unescape_bigram": html.unescape(data["title"]),
        "text_bigram": "".join(data["text"]) if isinstance(data["text"], list) else data["text"],
        "original_json": line if index_name == "hotpotqa" else json.dumps(data),
    }
    
    return "{}\n{}".format(
        json.dumps({"index": {"_id": "{}-{}".format(index_name, item["id"])}}), json.dumps(item)
    )

def generate_indexing_docs_from_bz2(bz2file, index_name, retrieval_method: str = "bm25"):
    """Preprocess hotpotqa corpus"""
    
    body = []
    with bz2.open(bz2file, "rt") as f:
        if retrieval_method == "bm25":
            body = [process_line(line, index_name) for line in f]
            body = "\n".join(body)
        else:
            for line in f:
                data = json.loads(line)
                item = {
                    "id": data["id"],
                    "title": data["title"],
                    "content": "".join(data["text"]),
                }
                body.append(item)

    return body

def preprocess_hotpotqa(input_data: str, retrieval_method: str = "bm25"):
    """Proprocess hotpotqa dataset."""
    filelist = glob(os.path.join(input_data, "*", "wiki_*.bz2"))
        
    pool = Pool()
    func = partial(generate_indexing_docs_from_bz2, 
               index_name="hotpotqa", 
               retrieval_method=retrieval_method)
    docs = list(
        tqdm(
            pool.imap(func, filelist), total=len(filelist)
        )
    )
    return docs

def preprocess_2wikimultihopqa(input_data: str, retrieval_method: str = "bm25"):
    """Preprocess 2wikimultihopqa dataset."""
    
    docs = []
    wikimultihop_train = json.load(open(os.path.join(input_data, "train.json")))
    wikimultihop_dev = json.load(open(os.path.join(input_data, "dev.json")))
    wikimultihop_test = json.load(open(os.path.join(input_data, "test.json")))
    
    wikimultihop_data = {}
    for item in tqdm(chain(wikimultihop_train, wikimultihop_dev, wikimultihop_test)):
        for title, sentences in item["context"]:
            para = " ".join(sentences)
            wikimultihop_data[para] = title
    wikimultihop_data_list = [
        {"id": i, "text": text, "title": title}
        for i, (text, title) in enumerate(wikimultihop_data.items())
    ]
    if retrieval_method == "bm25":
        for item in tqdm(wikimultihop_data_list):
            docs.append(process_line(item, "2wikimultihopqa"))
        return docs
    else:
        return wikimultihop_data_list

def preprocess_musique(input_data: str, retrieval_method: str = "bm25"):
    """preprocess musique dataset."""
    
    docs = []
    musique_train = [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_ans_v1.0_train.jsonl"))] + [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_full_v1.0_train.jsonl"))]
    musique_dev = [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_ans_v1.0_dev.jsonl"))] + [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_full_v1.0_dev.jsonl"))]
    musique_test = [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_ans_v1.0_test.jsonl"))] + [json.loads(line.strip()) for line in open(os.path.join(input_data, "musique_full_v1.0_test.jsonl"))]

    tot = 0
    musique_data = []
    hist = set()
    for item in tqdm(chain(musique_train, musique_dev, musique_test)):
        for p in item["paragraphs"]:
            stamp = p["title"] + " " + p["paragraph_text"]
            if not stamp in hist:
                musique_data.append(
                    {"id": tot, "text": p["paragraph_text"], "title": p["title"]}
                )
                hist.add(stamp)
                tot += 1
    
    if retrieval_method == "bm25":
        for item in tqdm(musique_data):
            docs.append(process_line(item, "musique"))
        return docs
    else:
        return musique_data

def preprocess_iirc(input_data: str, retrieval_method: str = "bm25"):
    """Preprocess iirc dataset."""
    
    with open(os.path.join(input_data, "context_articles.json")) as f:
        iirc_corpus = json.load(f)
    
    docs = []
    iirc_data = []
    
    for title, page_html in tqdm(iirc_corpus.items()):
        page_soup = BeautifulSoup(page_html, "html.parser")
        paragraph_texts = [
            text for text in page_soup.text.split("\n") if text.strip() and len(text.strip().split()) > 10
        ]

        # IIRC has a positional bias. 70% of the times, the first
        # is the supporting one, and almost all are in 1st 20.
        # So we scramble them to make it more challenging retrieval
        # problem.
        paragraph_indices_and_texts = [
            (paragraph_index, paragraph_text) for paragraph_index, paragraph_text in enumerate(paragraph_texts)
        ]
        random.shuffle(paragraph_indices_and_texts)
        for paragraph_index, paragraph_text in paragraph_indices_and_texts:
            id_ = hash_object(title + paragraph_text)
            es_para_obj = {
                "id": id_,
                "text": paragraph_text,
                "title": title,
            }
            iirc_data.append(es_para_obj)
    
    if retrieval_method == "bm25":
        for item in tqdm(iirc_data):
            docs.append(process_line(item, "iirc"))
        return docs
    else:
        return iirc_data

def split_kilt(input_file_path: str, output_file_path: str):
    # Initialize index
    idx = 0
    line_num=0

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                line_num += 1
                if line_num % 100000 == 0:
                    print("Finished Entity:",line_num)
                    print("Now 100-words paragraph:",idx)
                    print("=="*20)
                data = json.loads(line)
                text_list = data.get("text", [])
                title=data["wikipedia_title"]
                full_text = "".join(text_list)
                full_text = full_text.replace("BULLET::::","").replace("Section::::","")
                num_1=full_text.count("::::")
                num_2=full_text.count("print(a.split())")
                num_3=full_text.count("Section::::")
                if num_1!=num_2+num_3:
                    print(full_text)
                    print("=="*10)
            
                words = full_text.split()

                for i in range(0, len(words), 100):
                    paragraph = " ".join(words[i:i + 100])
                    paragraph = f"{idx}\t{title}   {paragraph}"
                    output_file.write(f"{paragraph}\n")
                    idx += 1
    print("kilt split finished.")
    
def preprocess_kilt(input_data: str):
    """Preprocess kilt for simpleqa dataset."""
    if not os.path.exists(os.path.join(input_data, "kilt_100_words.tsv")):
        split_kilt(os.path.join(input_data, "kilt_knowledgesource.json"), os.path.join(input_data, "kilt_100_words.tsv"))
    
    with open(os.path.join(input_data, "kilt_100_words.tsv"), 'r', encoding='utf-8') as file:
        corpus = file.readlines()
        c_len = len(corpus)
        docs = []
        line_num = 0
        for line in corpus:
            line_num += 1
            if line_num % 10000 == 0:
                print(f"Percent: {line_num}/{c_len}")

            title_text = line.split('\t')[1].strip('')
            docs.append(title_text)
    return docs
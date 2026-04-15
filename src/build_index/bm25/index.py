# Build index for hotpotqa, 2wikimultihopqa, musique and iirc using BM25
import sys
import os
import json

from argparse import ArgumentParser
from elasticsearch import Elasticsearch
from tqdm import tqdm

# load environment variables
project_root = os.environ.get("PROJECT_ROOT", "")
es_url = os.environ.get("ES_URL", "http://localhost:9200")

sys.path.append(os.path.join(project_root, "src", "build_index"))
from preprocess_datasets import preprocess_hotpotqa, preprocess_2wikimultihopqa, preprocess_musique, preprocess_iirc

# connect to elasticsearch engine
ES = Elasticsearch(hosts=es_url, timeout=500)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
    
def index_chunk(chunk, index_name, timeout="500s"):
    res = ES.bulk(index=index_name, body="\n".join(chunk), timeout=timeout)
    assert not res["errors"], res

def main(args):
    
    INDEX_NAME = args.index_name
    
    # create index
    if not args.dry:
        if ES.indices.exists(index=INDEX_NAME):
            ES.indices.delete(index=INDEX_NAME, ignore=[400, 403])
        else:
            ES.indices.create(
                index=INDEX_NAME,
                ignore=400,
                body=json.dumps(
                    {
                        "mappings": {
                            "doc": {
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "url": {"type": "keyword"},
                                    "title": {
                                        "type": "text",
                                        "analyzer": "simple",
                                        "copy_to": "title_all",
                                    },
                                    "title_unescape": {
                                        "type": "text",
                                        "analyzer": "simple",
                                        "copy_to": "title_all",
                                    },
                                    "text": {
                                        "type": "text",
                                        "analyzer": "my_english_analyzer",
                                    },
                                    "anchortext": {
                                        "type": "text",
                                        "analyzer": "my_english_analyzer",
                                    },
                                    "title_bigram": {
                                        "type": "text",
                                        "analyzer": "simple_bigram_analyzer",
                                        "copy_to": "title_all_bigram",
                                    },
                                    "title_unescape_bigram": {
                                        "type": "text",
                                        "analyzer": "simple_bigram_analyzer",
                                        "copy_to": "title_all_bigram",
                                    },
                                    "text_bigram": {
                                        "type": "text",
                                        "analyzer": "bigram_analyzer",
                                    },
                                    "anchortext_bigram": {
                                        "type": "text",
                                        "analyzer": "bigram_analyzer",
                                    },
                                    "original_json": {"type": "string"},
                                }
                            }
                        },
                        "settings": {
                            "analysis": {
                                "my_english_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_english_",
                                },
                                "simple_bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "shingle", "asciifolding"],
                                },
                                "bigram_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "stop",
                                        "shingle",
                                        "asciifolding",
                                    ],
                                },
                            },
                        },
                    }
                ),
            )
    
    # Load corpus corresponding to the dataset
    print(f"Loading {INDEX_NAME} docs...")
    docs = []
    if args.dry:
        docs = []
    else:
        if INDEX_NAME == "hotpotqa":
            docs = preprocess_hotpotqa(args.input_data, "bm25")
        elif INDEX_NAME == "2wikimultihopqa":
            docs = preprocess_2wikimultihopqa(args.input_data, "bm25")
        elif INDEX_NAME == "musique":
            docs = preprocess_musique(args.input_data, "bm25")
        elif INDEX_NAME == "iirc":
            docs = preprocess_iirc(args.input_data, "bm25")
        else:
            raise ValueError(f"Unknown index name: {INDEX_NAME}")
                
    count = sum(len(doc.split("\n")) for doc in docs) // 2
    
    if not args.dry:
        print("Indexing...")
        chunksize = 50
        for chunk in tqdm(
            chunks(docs, chunksize),
            total=(len(docs) + chunksize - 1) // chunksize,
        ):
            index_chunk(chunk, INDEX_NAME)
    print(f"{count} documents indexed in total.")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--index_name", type=str, required=True, default="hotpotqa", choices=["hotpotqa", "2wikimultihopqa", "musique", "iirc"], help="The name of the index")
    parser.add_argument("--input_data", type=str, help="The absolute path of the input data folder")
    parser.add_argument("--dry", action="store_true", help="Dry run")

    args = parser.parse_args()
    
    main(args)
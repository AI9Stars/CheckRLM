#!/bin/bash

set -euo pipefail

PROJECT_ROOT=""
TEMP_DIR="${PROJECT_ROOT}/.temp_download_corpus"

DATA_ROOT="${PROJECT_ROOT}/data"
HOTPOT_CORPUS="${DATA_ROOT}/hotpotqa/corpus"
WIKI_CORPUS="${DATA_ROOT}/2wikimultihopqa/corpus"
SIMPLEQA_CORPUS="${DATA_ROOT}/simpleqa/corpus"
MUSIQUE_CORPUS="${DATA_ROOT}/musique/corpus"
IIRC_CORPUS="${DATA_ROOT}/iirc/corpus"

pip install -q gdown 2>/dev/null || pip install gdown

mkdir -p "${TEMP_DIR}"
mkdir -p "${HOTPOT_CORPUS}" "${WIKI_CORPUS}" "${SIMPLEQA_CORPUS}" "${MUSIQUE_CORPUS}" "${IIRC_CORPUS}"

echo "Downloading 2wikimultihopqa corpus (train/dev/test json)..."
wget -q --show-progress "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1" \
  -O "${TEMP_DIR}/2wikimultihopqa.zip"
unzip -jo "${TEMP_DIR}/2wikimultihopqa.zip" -d "${WIKI_CORPUS}" -x "*.DS_Store"
find "${WIKI_CORPUS}" -maxdepth 1 -name '._*' -delete
shopt -s nullglob
for f in "${WIKI_CORPUS}"/*; do
  case "$(basename "${f}")" in
    id_aliases.json|train.json|dev.json|test.json) ;;
    *) rm -f "${f}" ;;
  esac
done
shopt -u nullglob

echo ""
echo "Downloading SimpleQA corpus (KILT knowledgesource json, may take a while)..."
wget -q --show-progress "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json" \
  -O "${SIMPLEQA_CORPUS}/kilt_knowledgesource.json"

echo ""
echo "Downloading musique corpus (v1.0 jsonl)..."
gdown "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" \
  -O "${TEMP_DIR}/musique_v1.0.zip"
unzip -jo "${TEMP_DIR}/musique_v1.0.zip" -d "${MUSIQUE_CORPUS}" -x "*.DS_Store"

echo ""
echo "Downloading IIRC Wikipedia corpus (context_articles, ~2–3 min)..."
wget -q --show-progress "https://iirc-dataset.s3.us-west-2.amazonaws.com/context_articles.tar.gz" \
  -O "${TEMP_DIR}/context_articles.tar.gz"
mkdir -p "${TEMP_DIR}/iirc_ctx"
tar -xzf "${TEMP_DIR}/context_articles.tar.gz" -C "${TEMP_DIR}/iirc_ctx"
IIRC_JSON="$(find "${TEMP_DIR}/iirc_ctx" -name "context_articles.json" -type f | head -1)"
if [[ -z "${IIRC_JSON}" ]]; then
  echo "ERROR: context_articles.json not found after extracting IIRC corpus." >&2
  exit 1
fi
mv -f "${IIRC_JSON}" "${IIRC_CORPUS}/context_articles.json"

echo ""
echo "Downloading HotpotQA Wikipedia corpus (enwiki abstracts, ~5–10 min)..."
wget -q --show-progress \
  "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2" \
  -O "${TEMP_DIR}/enwiki-abstracts.tar.bz2"
mkdir -p "${TEMP_DIR}/hotpot_unpack"
tar -xjf "${TEMP_DIR}/enwiki-abstracts.tar.bz2" -C "${TEMP_DIR}/hotpot_unpack"
ENWIKI_DIR="${TEMP_DIR}/hotpot_unpack/enwiki-20171001-pages-meta-current-withlinks-abstracts"
if [[ ! -d "${ENWIKI_DIR}" ]]; then
  echo "ERROR: expected directory not found after tar: ${ENWIKI_DIR}" >&2
  exit 1
fi
shopt -s dotglob nullglob
mv -f "${ENWIKI_DIR}"/* "${HOTPOT_CORPUS}/"
shopt -u dotglob nullglob
rmdir "${ENWIKI_DIR}" 2>/dev/null || true

rm -rf "${TEMP_DIR}"

echo ""
echo "Done. Corpus layout under ${DATA_ROOT}:"
if command -v tree >/dev/null 2>&1; then
  tree -L 2 --charset ascii \
    "${HOTPOT_CORPUS}" "${WIKI_CORPUS}" "${SIMPLEQA_CORPUS}" "${MUSIQUE_CORPUS}" "${IIRC_CORPUS}"
else
  for d in "${HOTPOT_CORPUS}" "${WIKI_CORPUS}" "${SIMPLEQA_CORPUS}" "${MUSIQUE_CORPUS}" "${IIRC_CORPUS}"; do
    echo ""
    echo "${d}"
    find "${d}" -maxdepth 2 | head -80
    echo "..."
  done
fi

echo ""
cat <<'EOF'
# Expected corpus roles (for --input_data when building BM25 index on raw dumps):
#
#   hotpotqa/corpus     -> directory containing */wiki_*.bz2 (from enwiki abstracts)
#   2wikimultihopqa/corpus -> train.json, dev.json, test.json, id_aliases.json
#   simpleqa/corpus       -> kilt_knowledgesource.json
#   musique/corpus      -> musique_*_v1.0_*.jsonl
#   iirc/corpus         -> context_articles.json
#
# The resulting data/*/corpus/ directories should look like:
# ├── 2wikimultihopqa/corpus   
# │   ├── id_aliases.json
# │   ├── dev.json
# │   ├── test.json
# │   └── train.json
# ├── simpleqa/corpus
# │   └── kilt_knowledgesource.json
# ├── hotpotqa/corpus
# │   ├── wiki_*.bz2
# ├── iirc/corpus
# │   └── context_articles.json
# └── musique/corpus
#     ├── dev_test_singlehop_questions_v1.0.json
#     ├── musique_ans_v1.0_*.jsonl
#     └── musique_full_v1.0_*.jsonl
EOF

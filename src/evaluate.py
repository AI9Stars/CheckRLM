import re
import string
from collections import Counter
from utils import load_jsonl, save_json
from tqdm import tqdm

def acc_score(predictions, answers):
    num_correct = 0
    acc_scores = []
    for id, answer in enumerate(answers):
        pred = predictions[id]
        correctness = (
            "True" if any(ans.lower() in pred.lower() for ans in answer) else "False"
        )
        if correctness == "True":
            num_correct += 1
            acc_score = 1.0
        else:
            acc_score = 0.0
        acc_scores.append(acc_score)
    acc = num_correct / len(answers)
    return acc_scores, round(acc * 100, 2)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(predictions, answers):
    total_score = 0.0
    em_scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                int(normalize_answer(prediction) == normalize_answer(ground_truth)),
            )
        total_score += score
        em_scores.append(score)
    
    return em_scores, round(100 * total_score / len(predictions), 2)

def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def F1_scorer(predictions, answers):
    total_score = 0.0
    f1_scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
        f1_scores.append(score)
    return f1_scores, round(100 * total_score / len(predictions), 2)

def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False
    
def post_process_answer(predicted_answer: str):
    if r"\boxed" in predicted_answer:
        pattern = r"\\boxed{((?:[^{}]|\\text{[^{}]*})*)}"
        matches = re.findall(pattern, predicted_answer)
        pred_answer = []
        for match in matches:
            # 移除 \text{} 标记，保留内容
            cleaned_match = re.sub(r'\\text\{([^}]*)\}', r'\1', match)
            pred_answer.append(cleaned_match)
        pred_answer = " ".join(pred_answer)
    elif "Answer" in predicted_answer:
        pred_answer = predicted_answer.split("Answer")[-1].replace(":", "").replace("*", "").strip()
    else:
        pred_answer = predicted_answer.replace("\n\n", "")
        
    return pred_answer

def eval(
    pred_path: str,
    eval_result_file: str = "eval_data.json"
):
    predictions = load_jsonl(pred_path)
    
    ground_truths = []
    pred_answers = []
    for item in tqdm(predictions):
        if isinstance(item["ground_truth"], str):
            ground_truths.append([item["ground_truth"]])
        else:
            ground_truths.append(item["ground_truth"])
        pred_answers.append(post_process_answer(item["predicted_answer"]))
    
    em_scores, em = compute_exact(pred_answers, ground_truths)
    f1_scores, f1 = F1_scorer(pred_answers, ground_truths)
    acc_scores, acc = acc_score(pred_answers, ground_truths)
    
    with open(pred_path.replace("outputs.jsonl", "eval_results.txt"), "w") as file:
        file.write(f"f1: {f1}\nem: {em}\nacc: {acc}")
    
    results = []
    for data, pred, em, f1, acc in zip(predictions, pred_answers, em_scores, f1_scores, acc_scores):
        result_obj = {
            "question_id": data["question_id"],
            "question_text": data["question_text"],
            "pred_answer": pred,
            "ground_truth": data["ground_truth"],
            "em": em,
            "f1": f1,
            "acc": acc,
        }
        results.append(result_obj)
    
    results_path = pred_path.replace("outputs.jsonl", eval_result_file)
    save_json(results_path, results)
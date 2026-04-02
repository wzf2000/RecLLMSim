import os
import json
from argparse import ArgumentParser

from lib.utils import set_seed
from lib.evaluation import evaluate
from lib.llm import generate, get_messages
from legacy.ml import evaluate_ml, train_predict_ml
from legacy.satisfaction_data import get_data, get_USS_data
from legacy.reason_data import get_reason_data, get_reason_data2, gt_map_reverse
from legacy.merge_data import get_merge_data
from legacy.prompts import binary_prompt, reason_descriptions, reflect_prompts, refine_prompts, compare_prompts

def work_ml(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, binary: bool = False, profile: bool = False, split: str | None = None) -> tuple[float, float]:
    set_seed(42)
    if split is not None:
        test_data, train_data = get_USS_data(split, sample=sample, training=True, binary=binary)
    else:
        test_data, train_data = get_data(sample=sample, training=True, binary=binary)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name, profile=profile)

def work_ml_reason(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, profile: bool = False, data_version: int = 1) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_reason_data(sample=sample, training=True) if data_version == 1 else get_reason_data2(sample=sample, training=True)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name, profile=profile)

def work_ml_merged(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, profile: bool = False) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_merge_data(sample=sample, training=True)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name, profile=profile)

def reason_test(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, binary: bool = False, profile: bool = False, data_version: int = 1, split: str | None = None) -> list[dict]:
    assert split is not None, "Split must be provided for USS data."
    set_seed(42)
    _, train_data = get_reason_data(sample=sample, training=True) if data_version == 1 else get_reason_data2(sample=sample, training=True)
    USS_data, _ = get_USS_data(split, sample=sample, training=True, binary=True)
    USS_data = [data for data in USS_data if data['ground_truth'] == 0]  # Filter for negative samples only
    predictions = train_predict_ml(train_data, USS_data, vectorizer=vectorizer, model_name=model_name, profile=profile)
    ret = [{
        'input': data['history'],
        'user_response': data['user_response'],
        'satisfaction': data['satisfaction'],
        'prediction': gt_map_reverse[int(pred)],
    } for data, pred in zip(USS_data, predictions)]
    with open(f'results/{model_name}_USS_{split}_predictions.json', 'w') as f:
        json.dump(ret, f, ensure_ascii=False, indent=4)
    print(f"Predictions saved to results/{model_name}_USS_predictions.json")
    return ret

def refine_test(model: str, version: int, prompt_version: int, sample: bool = True, data_version: int = 2):
    import random
    from tqdm import tqdm
    set_seed(42)
    test_data, examples = get_reason_data(sample=sample, training=False) if data_version == 1 else get_reason_data2(sample=sample, training=False)
    # output_file = f'results/{model}_refine{data_version}_predictions_v{version}{"_sampled" if sample else ""}.json'
    better = 0
    worse = 0
    error = 0
    sample_size = 100
    print(len(test_data))
    # filter data to only include those with satisfaction < 3
    test_data = [data for data in test_data if data['satisfaction'] < 3]
    print(len(test_data))
    for data in tqdm(test_data[:sample_size]):
        suggestion = generate(get_messages(reflect_prompts[prompt_version].format(context=data['history'], reason=reason_descriptions[data['ground_truth']])), model)
        refined_output = generate(get_messages(refine_prompts[prompt_version].format(context=data['history'], suggestion=suggestion)), model)
        tag = 0
        if random.random() < 0.5:
            compare_output = generate(get_messages(compare_prompts[prompt_version].format(context=data['history'], answer1=refined_output, answer2=data['origin_response'])), model)
            if compare_output.strip()[0] == '1':
                better += 1
            elif compare_output.strip()[0] == '2':
                worse += 1
                tag = 1
            else:
                error += 1
        else:
            compare_output = generate(get_messages(compare_prompts[prompt_version].format(context=data['history'], answer1=data['origin_response'], answer2=refined_output)), model)
            if compare_output.strip()[0] == '2':
                better += 1
            elif compare_output.strip()[0] == '1':
                worse += 1
                tag = 1
            else:
                error += 1
        if tag == 1:
            print(f"Original Response: {data['origin_response']}")
            print(f"Refined Response: {refined_output}")
            print(f"Suggestion: {suggestion}")
            print(f"Compare Output: {compare_output}")
            print(f"Ground Truth: {data['ground_truth']}")
            print(f"History: {data['history']}")
            print("-" * 50)
    print("Refinement Test Results:")
    print(f"Better: {better}, Worse: {worse}, Error: {error}")
    print(f"Refinement Test: {better} out of {len(test_data[:sample_size])} cases were improved, better rate: {better / len(test_data[:sample_size]):.4f}")

def evaluate_dir(dir_path: str, sample: bool = True, data_version: int = 1) -> float:
    set_seed(42)
    test_data, train_data = get_reason_data(sample=sample, training=True) if data_version == 1 else get_reason_data2(sample=sample, training=True)
    acc = []
    for i, data in enumerate(test_data):
        gt = data['ground_truth']
        with open(f'{dir_path}/{i}.txt', 'r') as f:
            lines = f.readlines()
            # get the last non-empty line
            for line in lines[::-1]:
                if line.strip():
                    break
            prediction = line.strip().split(',')
            prediction = [int(p) for p in prediction]
        if gt in prediction:
            acc.append(1)
        else:
            acc.append(0)
    acc = sum(acc) / len(acc)
    print(f"Accuracy: {acc:.4f}")
    return acc

def output_data(data_version: int = 1, reason: bool = True):
    set_seed(42)
    if reason:
        test_data, train_data = get_reason_data(sample=False, training=True) if data_version == 1 else get_reason_data2(sample=False, training=True)
        output_dir = f'dataset/reasoning_data_v{data_version}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/train.json', 'w') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        with open(f'{output_dir}/test.json', 'w') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
    else:
        test_data, train_data = get_data(sample=False, training=True, binary=True)
        test_data, examples = get_data(sample=False, training=True, binary=True)
        output_dir = 'dataset/satisfaction_data'
        os.makedirs(output_dir, exist_ok=True)
        conversations = []
        prompt = binary_prompt
        for data in train_data:
            instruction = prompt.format(context=data['history'], zero_example=examples[0], one_example=examples[1], task_context=data['task_context'])
            conversations.append({
                "messages": [
                    {
                        'role': 'system',
                        'content': 'You are a skilled conversational analyst.'
                    },
                    {
                        'role': 'user',
                        'content': instruction
                    },
                    {
                        'role': 'assistant',
                        'content': str(data['ground_truth'])
                    }
                ]
            })
        with open(f'{output_dir}/RecLLMSim.json', 'w') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)

def parse_args():
    parser = ArgumentParser(description="Evaluate models")
    parser.add_argument("--pipe", type=str, default="train", choices=["train", "eval_llm", "output_data", "reason_test", "refine_test"], help="Pipeline to run")
    parser.add_argument("--split", type=str, default=None, help="Split for train & test, e.g., CCPE, JDDC")
    parser.add_argument("--reason", action="store_true", help="Use reasoning data")
    parser.add_argument("--merge", action="store_true", help="Use merged data")
    parser.add_argument("--data_version", type=int, default=1, help="Data version for reasoning data")
    parser.add_argument("-m", "--model", type=str, help="Model name")
    parser.add_argument("--vectorizer", type=str, choices=["count", "tfidf"], default="tfidf", help="Vectorizer to use for ML models")
    parser.add_argument("-s", "--sample", action="store_true", help="Use sampled data")
    parser.add_argument("-b", "--binary", action="store_true", help="Use binary classification for satisfaction data")
    parser.add_argument("--regression", action="store_true", help="Use regression for LM models & satisfaction data")
    parser.add_argument("-v", "--version", type=int, default=1, help="Version of the experiment for LLM models")
    parser.add_argument("--prompt_version", type=int, default=0, help="Prompt version for LLM models & reason data")
    parser.add_argument("--in_context", action="store_true", help="Use in-context learning for LLM models")
    parser.add_argument("--profile", action="store_true", help="Use profile for ML & LM models")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.pipe == "train":
        if args.reason:
            work_ml_reason(args.vectorizer, args.model, sample=args.sample, profile=args.profile, data_version=args.data_version)
        elif args.merge:
            work_ml_merged(args.vectorizer, args.model, sample=args.sample, profile=args.profile)
        else:
            work_ml(args.vectorizer, args.model, sample=args.sample, binary=args.binary, profile=args.profile, split=args.split)
    elif args.pipe == "eval_llm":
        evaluate_dir(f'results/{args.model}_reason{args.data_version}_predictions_v{args.version}{"_sampled" if args.sample else ""}_cache', sample=args.sample, data_version=args.data_version)
    elif args.pipe == "output_data":
        output_data(data_version=args.data_version, reason=args.reason)
    elif args.pipe == "reason_test":
        reason_test(args.vectorizer, args.model, sample=args.sample, binary=args.binary, profile=args.profile, data_version=args.data_version, split=args.split)
    elif args.pipe == "refine_test":
        refine_test(args.model, args.version, args.prompt_version, sample=args.sample, data_version=args.data_version)
    else:
        raise ValueError(f"Unknown pipe: {args.pipe}")

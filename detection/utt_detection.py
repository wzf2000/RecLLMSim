import os
import json
from argparse import ArgumentParser

from ml import evaluate_ml
from lm import evaluate_lm
from llm import predict_llm, predict_llm_in_context
from utils import set_seed
from satisfaction_data import get_data
from reason_data import get_reason_data, get_reason_data2
from evaluation import evaluate
from prompts import score_prompt, binary_prompt, reason_prompts, reason_in_context_prompts

def work_llm(model: str, version: int, sample: bool = False, binary: bool = False) -> tuple[float, float]:
    set_seed(42)
    test_data, examples = get_data(sample=sample, training=False, binary=binary)
    ground_truth = [data['ground_truth'] for data in test_data]
    output_file = f'results/{model}_predictions_v{version}{"_sampled" if sample else ""}{"_binary" if binary else ""}.json'
    prompt = binary_prompt if binary else score_prompt
    if binary:
        prompts = [prompt.format(context=data['history'], task_context=data['task_context'], zero_example=examples[0], one_example=examples[1]) for data in test_data]
    else:
        prompts = [prompt.format(context=data['history'], task_context=data['task_context']) for data in test_data]
    predictions = predict_llm(prompts, model, output_file=output_file)
    return evaluate(predictions, ground_truth)

def work_ml(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, binary: bool = False, profile: bool = False) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_data(sample=sample, training=True, binary=binary)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name, profile=profile)

def work_lm(model_name: str, regression: bool = False, sample: bool = True, binary: bool = False, profile: bool = False) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_data(sample=sample, training=True, binary=binary)
    results = evaluate_lm(model_name, train_data, test_data, num_labels=2 if binary else 5, regression=regression, profile=profile)
    return results['eval_mse'], results['eval_rmse']

def work_llm_reason(model: str, version: int, prompt_version: int, sample: bool = False, in_context: bool = False, data_version: int = 1) -> tuple[float, float]:
    set_seed(42)
    test_data, examples = get_reason_data(sample=sample, training=False) if data_version == 1 else get_reason_data2(sample=sample, training=False)
    ground_truth = [data['ground_truth'] for data in test_data]
    output_file = f'results/{model}_reason{data_version}_predictions_v{version}{"_sampled" if sample else ""}.json'
    if not in_context:
        prompt = reason_prompts[prompt_version]
        if '{profile}' in prompt:
            prompts = [prompt.format(context=data['history'], profile=data['profile'], zero_example=examples[0], one_example=examples[1], two_example=examples[2], three_example=examples[3]) for data in test_data]
        else:
            prompts = [prompt.format(context=data['history'], zero_example=examples[0], one_example=examples[1], two_example=examples[2], three_example=examples[3]) for data in test_data]
        predictions = predict_llm(prompts, model, output_file=output_file)
    else:
        prompt = reason_in_context_prompts[prompt_version]
        histories = [data['origin_history'] for data in test_data]
        if '{profile}' in prompt:
            prompts = [prompt.format(zero_example=examples[0], one_example=examples[1], two_example=examples[2], three_example=examples[3], profile=data['profile']) for data in test_data]
        else:
            prompts = [prompt.format(zero_example=examples[0], one_example=examples[1], two_example=examples[2], three_example=examples[3]) for _ in test_data]
        predictions = predict_llm_in_context(prompts, histories, model, output_file=output_file)
    return evaluate(predictions, ground_truth)

def work_ml_reason(vectorizer: str = 'tfidf', model_name: str = 'RF', sample: bool = True, profile: bool = False, data_version: int = 1) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_reason_data(sample=sample, training=True) if data_version == 1 else get_reason_data2(sample=sample, training=True)
    return evaluate_ml(train_data, test_data, vectorizer=vectorizer, model_name=model_name, profile=profile)

def work_lm_reason(model_name: str, sample: bool = True, profile: bool = False, data_version: int = 1) -> tuple[float, float]:
    set_seed(42)
    test_data, train_data = get_reason_data(sample=sample, training=True) if data_version == 1 else get_reason_data2(sample=sample, training=True)
    results = evaluate_lm(model_name, train_data, test_data, num_labels=5, regression=False, profile=profile)
    return results['eval_mse'], results['eval_rmse']

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

def output_data(data_version: int = 1):
    set_seed(42)
    test_data, train_data = get_reason_data(sample=False, training=True) if data_version == 1 else get_reason_data2(sample=False, training=True)
    output_dir = f'dataset/reasoning_data_v{data_version}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

def parse_args():
    parser = ArgumentParser(description="Evaluate models")
    parser.add_argument("-t", "--type", type=str, choices=["ml", "lm", "llm", "eval_llm", "output_data"], required=True, help="Type of model to evaluate")
    parser.add_argument("--reason", action="store_true", help="Use reasoning data")
    parser.add_argument("--data_version", type=int, default=1, help="Data version for reasoning data")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
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
    if args.type == "ml":
        if args.reason:
            work_ml_reason(args.vectorizer, args.model, sample=args.sample, profile=args.profile, data_version=args.data_version)
        else:
            work_ml(args.vectorizer, args.model, sample=args.sample, binary=args.binary, profile=args.profile)
    elif args.type == "lm":
        if args.reason:
            work_lm_reason(args.model, sample=args.sample, profile=args.profile, data_version=args.data_version)
        else:
            work_lm(args.model, regression=args.regression, sample=args.sample, binary=args.binary, profile=args.profile)
    elif args.type == "llm":
        if args.reason:
            work_llm_reason(args.model, args.version, args.prompt_version, sample=args.sample, in_context=args.in_context, data_version=args.data_version)
        else:
            work_llm(args.model, args.version, sample=args.sample, binary=args.binary)
    elif args.type == "eval_llm":
        evaluate_dir(f'results/{args.model}_reason{args.data_version}_predictions_v{args.version}{"_sampled" if args.sample else ""}_cache', sample=args.sample, data_version=args.data_version)
    elif args.type == "output_data":
        output_data(data_version=args.data_version)

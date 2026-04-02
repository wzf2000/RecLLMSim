#! /bin/bash

# Example usage:
## output data
python utt_detection.py -t output_data --data_version 1
python utt_detection.py -t output_data --data_version 2

## binary satisfaction data
python utt_detection.py -t ml -m RF --vectorizer count -b
python utt_detection.py -t ml -m RF --vectorizer tfidf -b
python utt_detection.py -t ml -m XGB --vectorizer count -b
python utt_detection.py -t ml -m XGB --vectorizer tfidf -b
python utt_detection.py -t lm -m bert-base-chinese -b
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 1 -b

## scoring satisfaction data
python utt_detection.py -t ml -m RF --vectorizer count
python utt_detection.py -t ml -m RF --vectorizer tfidf
python utt_detection.py -t ml -m XGB --vectorizer count
python utt_detection.py -t ml -m XGB --vectorizer tfidf
python utt_detection.py -t ml -m RFReg --vectorizer count
python utt_detection.py -t ml -m RFReg --vectorizer tfidf
python utt_detection.py -t ml -m XGBReg --vectorizer count
python utt_detection.py -t ml -m XGBReg --vectorizer tfidf
python utt_detection.py -t lm -m bert-base-chinese
python utt_detection.py -t lm -m bert-base-chinese --regression
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 1 -s
python utt_detection.py -t llm -m gpt-4-turbo-preview -v 1 -s
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 1 -s
python utt_detection.py -t llm -m claude-3-5-sonnet-20241022 -v 1 -s
python utt_detection.py -t llm -m deepseek-r1 -v 1 -s

## reason data V1
python utt_detection.py -t ml -m RF --vectorizer count --reason --data_version 1
python utt_detection.py -t ml -m RF --vectorizer tfidf --reason --data_version 1
python utt_detection.py -t ml -m XGB --vectorizer count --reason --data_version 1
python utt_detection.py -t ml -m XGB --vectorizer tfidf --reason --data_version 1
python utt_detection.py -t lm -m bert-base-chinese --reason --data_version 1
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 1 --prompt_version 0 --data_version 1
python utt_detection.py -t llm -m deepseek-r1 -v 1 --prompt_version 0 --data_version 1
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 4 --prompt_version 0 --in_context --data_version 1
python utt_detection.py -t llm -m deepseek-r1 -v 5 --prompt_version 1 --data_version 1
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 6 --prompt_version 1 --in_context --data_version 1
python utt_detection.py -t llm -m deepseek-r1 -v 7 --prompt_version 2 --data_version 1

## reason data V2
python utt_detection.py -t ml -m RF --vectorizer count --reason --data_version 2
python utt_detection.py -t ml -m RF --vectorizer tfidf --reason --data_version 2
python utt_detection.py -t ml -m XGB --vectorizer count --reason --data_version 2
python utt_detection.py -t ml -m XGB --vectorizer tfidf --reason --data_version 2
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 8 --prompt_version 0 --in_context --data_version 2
python utt_detection.py -t llm -m deepseek-r1 -v 9 --prompt_version 1 --data_version 2
python utt_detection.py -t llm -m gpt-4o-2024-08-06 -v 10 --prompt_version 2 --in_context --data_version 2
python utt_detection.py -t llm -m deepseek-r1 -v 11 --prompt_version 3 --data_version 2
python utt_detection.py -t llm -m deepseek-r1 -v 12 --prompt_version 3 --data_version 2
python utt_detection.py -t llm -m deepseek-reasoner -v 13 --prompt_version 3 --data_version 2
python utt_detection.py -t llm -m deepseek-r1 -v 14 --prompt_version 3 --data_version 2
python utt_detection.py -t llm -m deepseek-r1 -v 15 --prompt_version 4 --data_version 2
python utt_detection.py -t eval_llm -m deepseek-r1 -v 15 --data_version 2

python utt_detection.py -t llm -m llama3.1-8b -v 9 --prompt_version 1 --data_version 2
python utt_detection.py -t llm -m qwen2.5-7b-instruct -v 9 --prompt_version 1 --data_version 2
python utt_detection.py -t llm -m qwen2.5-7b-instruct -v 8 --prompt_version 0 --data_version 2
python utt_detection.py -t llm -m qwen2.5-7b-instruct -v 11 --prompt_version 3 --data_version 2
python utt_detection.py -t llm -m qwen2.5-7b-instruct -v 15 --prompt_version 4 --data_version 2
python utt_detection.py -t eval_llm -m qwen2.5-7b-instruct -v 15 --data_version 2

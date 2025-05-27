#! /bin/bash

python predict-llm.py -m gpt-4o-2024-08-06 -t sim -l zh
python predict-llm.py -m gpt-4o-2024-08-06 -t human
python predict-llm.py -m gpt-4o-2024-08-06 -t sim2human
python predict-llm.py -m gpt-4o-2024-08-06 -t human2sim
python predict-llm.py -m gpt-4o-2024-08-06 -t sim2human2
python predict-llm.py -m gpt-4o-2024-08-06 -t human2sim2

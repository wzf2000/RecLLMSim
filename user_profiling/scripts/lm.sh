#! /bin/bash

BERT_BASE=bert-base-chinese

python predict-lm.py -m $BERT_BASE -t sim -l zh
python predict-lm.py -m $BERT_BASE -t sim2human
python predict-lm.py -m $BERT_BASE -t sim4human
python predict-lm.py -m $BERT_BASE -t human2sim
python predict-lm.py -m $BERT_BASE -t sim2human2
python predict-lm.py -m $BERT_BASE -t sim4human2
python predict-lm.py -m $BERT_BASE -t human2sim2
python predict-lm.py -m $BERT_BASE -t sim4human3
python predict-lm.py -m $BERT_BASE -t sim2human3
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.1
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.2
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.5
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 1
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 2
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 5
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 10

python predict-lm.py -m $BERT_BASE -t human
python predict-lm.py -m $BERT_BASE -t human -d 2
python predict-lm.py -m $BERT_BASE -t sim2human3 -d 2

python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.1 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.2 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 0.5 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human4 -r 1 -d 2

python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc hot --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc cold --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc hot --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc cold --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc hot --topk 5
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc cold --topk 5
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc hot --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc cold --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc hot --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc cold --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc hot --topk 5
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc cold --topk 5
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc hot --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc cold --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc hot --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc cold --topk 1
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc hot --topk 5
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc cold --topk 5

python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc hot --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc cold --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc both --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc hot --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc cold --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc both --topk 3
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc hot --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc cold --topk 6
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc both --topk 3

python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc hot --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc cold --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.1 -hc both --topk 3 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc hot --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc cold --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 0.5 -hc both --topk 3 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc hot --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc cold --topk 6 -d 2
python predict-lm.py -m $BERT_BASE -t sim4human5 -r 1 -hc both --topk 3 -d 2

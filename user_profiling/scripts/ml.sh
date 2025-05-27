#! /bin/bash

python predict-ml.py -m "XGB-MultiOutput" -t sim -l zh
python predict-ml.py -m "XGB-ClassifierChain" -t sim -l zh
python predict-ml.py -m "RF-MultiOutput" -t sim -l zh
python predict-ml.py -m "RF-ClassifierChain" -t sim -l zh

python predict-ml.py -m "XGB-MultiOutput" -t human
python predict-ml.py -m "XGB-ClassifierChain" -t human
python predict-ml.py -m "RF-MultiOutput" -t human
python predict-ml.py -m "RF-ClassifierChain" -t human

python predict-ml.py -m "XGB-MultiOutput" -t human2sim
python predict-ml.py -m "XGB-ClassifierChain" -t human2sim
python predict-ml.py -m "RF-MultiOutput" -t human2sim
python predict-ml.py -m "RF-ClassifierChain" -t human2sim

python predict-ml.py -m "XGB-MultiOutput" -t human
python predict-ml.py -m "XGB-ClassifierChain" -t human
python predict-ml.py -m "RF-MultiOutput" -t human
python predict-ml.py -m "RF-ClassifierChain" -t human

python predict-ml.py -m "XGB-MultiOutput" -t human -d 2
python predict-ml.py -m "XGB-ClassifierChain" -t human -d 2
python predict-ml.py -m "RF-MultiOutput" -t human -d 2
python predict-ml.py -m "RF-ClassifierChain" -t human -d 2

python predict-ml.py -m "XGB-MultiOutput" -t sim4human
python predict-ml.py -m "XGB-ClassifierChain" -t sim4human
python predict-ml.py -m "RF-MultiOutput" -t sim4human
python predict-ml.py -m "RF-ClassifierChain" -t sim4human

python predict-ml.py -m "XGB-MultiOutput" -t human2sim
python predict-ml.py -m "XGB-ClassifierChain" -t human2sim
python predict-ml.py -m "RF-MultiOutput" -t human2sim
python predict-ml.py -m "RF-ClassifierChain" -t human2sim

python predict-ml.py -m "XGB-MultiOutput" -t sim2human2
python predict-ml.py -m "XGB-ClassifierChain" -t sim2human2
python predict-ml.py -m "RF-MultiOutput" -t sim2human2
python predict-ml.py -m "RF-ClassifierChain" -t sim2human2

python predict-ml.py -m "XGB-MultiOutput" -t sim4human2
python predict-ml.py -m "XGB-ClassifierChain" -t sim4human2
python predict-ml.py -m "RF-MultiOutput" -t sim4human2
python predict-ml.py -m "RF-ClassifierChain" -t sim4human2

python predict-ml.py -m "XGB-MultiOutput" -t human2sim2
python predict-ml.py -m "XGB-ClassifierChain" -t human2sim2
python predict-ml.py -m "RF-MultiOutput" -t human2sim2
python predict-ml.py -m "RF-ClassifierChain" -t human2sim2

python predict-ml.py -m "XGB-MultiOutput" -t sim4human3
python predict-ml.py -m "XGB-ClassifierChain" -t sim4human3
python predict-ml.py -m "RF-MultiOutput" -t sim4human3
python predict-ml.py -m "RF-ClassifierChain" -t sim4human3

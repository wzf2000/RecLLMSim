#!/bin/bash
set -e

# 枚举 LR, RF
for model in LR RF XGB GNB CNB MNB SVM; do
  # 枚举 task_name
  for task_name in travel gift recipe skill; do
    # 直接运行不增强的基线
    echo "python next_intent_prediction.py --data_task_name ${task_name} -t human ml -m $model"
    python next_intent_prediction.py --data_task_name ${task_name} -t human ml -m $model
    # 枚举增强方法
    for aug_method in none hot_3 hot_6 cold_3 cold_6 ratio; do
    # for aug_method in none; do
    # 枚举数据增强比例
      for ratio in 0.1 0.2 0.5 1.0; do
        if [ "$aug_method" != "none" ] && [ "$ratio" = "1.0" ]; then
          continue
        fi
        if [ "$aug_method" == "cold_3" ] && [ "$ratio" != "0.1" ]; then
          continue
        fi
        if [ "$aug_method" = "none" ]; then
          echo "python next_intent_prediction.py --data_task_name ${task_name} -t aug_$ratio ml -m $model"
          python next_intent_prediction.py --data_task_name ${task_name} -t aug_$ratio ml -m $model
        else
          echo "python next_intent_prediction.py --data_task_name ${task_name} -t aug_${aug_method}_$ratio ml -m $model"
          python next_intent_prediction.py --data_task_name ${task_name} -t aug_${aug_method}_$ratio ml -m $model
        fi
      done
    done
  done
done

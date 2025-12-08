#!/bin/bash
set -e

# 枚举 LR, RF
for model in LR RF XGB GNB CNB MNB SVM; do
  # 枚举增强方法
  for aug_method in none hot_3 hot_6 cold_3 cold_6 ratio; do
  # 枚举数据增强比例
    for ratio in 0.1 0.2 0.5 1.0; do
      if [ "$aug_method" != "none" ] && [ "$ratio" = "1.0" ]; then
        continue
      fi
      if [ "$aug_method" == "cold_3" ] && [ "$ratio" != "0.1" ]; then
        continue
      fi
      if [ "$aug_method" = "none" ]; then
        echo "python next_intent_prediction.py -t aug_$ratio --filter ml -m $model"
        python next_intent_prediction.py -t aug_$ratio --filter ml -m $model
      else
        echo "python next_intent_prediction.py -t aug_${aug_method}_$ratio --filter ml -m $model"
        python next_intent_prediction.py -t aug_${aug_method}_$ratio --filter ml -m $model
      fi
    done
  done
done

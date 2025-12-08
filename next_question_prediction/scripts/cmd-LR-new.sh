#!/bin/bash
set -e

# 枚举 LR, RF
for model in LR-new; do
  # 运行不增强的基线
  echo "python next_intent_prediction.py -t human ml -m $model"
  python next_intent_prediction.py -t human ml -m $model
  # 枚举增强方法
  for aug_method in none hot_3 hot_6 cold_3 cold_6 ratio; do
  # 枚举数据增强比例
    for ratio in 0.1 0.2 0.5; do
      if [ "$aug_method" != "none" ] && [ "$ratio" = "1.0" ]; then
        continue
      fi
      if [ "$aug_method" == "cold_3" ] && [ "$ratio" != "0.1" ]; then
        continue
      fi
      if [ "$aug_method" = "none" ]; then
        echo "python next_intent_prediction.py -t augv2_$ratio ml -m $model"
        python next_intent_prediction.py -t augv2_$ratio ml -m $model
      else
        echo "python next_intent_prediction.py -t augv2_${aug_method}_$ratio ml -m $model"
        python next_intent_prediction.py -t augv2_${aug_method}_$ratio ml -m $model
      fi
    done
  done
done

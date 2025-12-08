#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=next_intent_prediction

for task_name in travel gift recipe skill; do
  # 直接运行不增强的基线
  python next_intent_prediction.py --data_task_name ${task_name} -t human lm -m bert-base-chinese
  # 枚举增强方法
  for aug_method in none hot_3 hot_6 cold_3 cold_6 ratio; do
    # 枚举数据增强比例
    for ratio in 0.1 0.2 0.5 1.0; do
      if [ "$aug_method" != "none" ] && [ "$ratio" = "1.0" ]; then
        continue
      fi
      if [ "$aug_method" = "none" ]; then
        python next_intent_prediction.py --data_task_name ${task_name} -t aug_$ratio lm -m bert-base-chinese
      else
        python next_intent_prediction.py --data_task_name ${task_name} -t aug_${aug_method}_$ratio lm -m bert-base-chinese
      fi
    done
  done
done

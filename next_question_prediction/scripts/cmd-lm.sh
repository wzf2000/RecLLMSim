#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=8
export WANDB_PROJECT=next_intent_prediction

# 枚举增强方法
for aug_method in none hot_3 hot_6 cold_3 cold_6 ratio; do
  # 枚举数据增强比例
  for ratio in 0.1 0.2 0.5 1.0; do
    if [ "$aug_method" != "none" ] && [ "$ratio" = "1.0" ]; then
      continue
    fi
    if [ "$aug_method" = "none" ]; then
      echo "python next_intent_prediction.py -t aug_$ratio lm -m bert-base-chinese"
      python next_intent_prediction.py -t aug_$ratio lm -m bert-base-chinese
    else
      echo "python next_intent_prediction.py -t aug_${aug_method}_$ratio lm -m bert-base-chinese"
      python next_intent_prediction.py -t aug_${aug_method}_$ratio lm -m bert-base-chinese
    fi
  done
done

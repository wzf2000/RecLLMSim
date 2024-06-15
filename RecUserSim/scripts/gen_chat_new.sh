#! /bin/bash

CONFIG=config/gen_chat_new.json

python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG  --task_type "new travel planning" --env_id 3 --context_ids 0 --preferences 50

python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG  --task_type "new travel planning" --env_id 3 --context_ids 5 --preferences 50

python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG  --task_type "new travel planning" --env_id 3 --context_ids 7 --preferences 50

python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG  --task_type "new travel planning" --env_id 3 --context_ids 9 --preferences 50

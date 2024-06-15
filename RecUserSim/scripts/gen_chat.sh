#! /bin/bash

CONFIG=config/gen_chat_en.json

# en whole
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "travel planning" --context_ids 1,2,5,7,8,11 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "skills learning planning" --context_ids 0,7,9 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "choosing restaurant" --context_ids 3,4,8 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "preparing gifts" --context_ids 1,5,6,10 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "recipe planning" --context_ids 0,3,7 --env_id 1

# en update
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "skills learning planning" --context_ids 2,4 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "choosing restaurant" --context_ids 1 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "preparing gifts" --context_ids 4 --env_id 1
python main.py --mode gen_chat --verbose SUCCESS -c $CONFIG --task_type "recipe planning" --context_ids 4 --env_id 1

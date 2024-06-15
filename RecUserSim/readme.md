# Recommendation User Simulation

## Files

- `main.py`: The program entrance of task-based recommendation conversation generation tasks.
- `scripts`: Contains generation script.
- `config`: Contains configuration files for generation and experiments.
- `data`: Contains the task context settings and generated user preferences.
- `usim`: Source code folder.

## Usage

Add `config/configV4.json` as following:

```json
{
    "api_base": "API_BASE, looks like https://..../v1, optional",
    "api_key": "Your API key",
    "engine": "The chat model to use, e.g. gpt-3.5-turbo, gpt-4-turbo-preview, etc.",
    "temperature": 0.5 // The temperature used in the chat completions.
}
```

For task-based recommendation conversation generation:

```shell
# User profile generation
# Change POOL_SIZE to control the generation number
python main.py --task preference --config config/preference_gen_en.json --pool_size $POOL_SIZE

# Conversation generation
# Set the task type in TASK_TYPE, e.g. TASK_TYPE="travel planning"
# Determine the specific context by set the CONTEXT_IDS, e.g. CONTEXT_IDS=[0,2,4]
python main.py --task gen_chat --config config/gen_chat_en.json --task_type $TASK_TYPE --context_ids $CONTEXT_IDS --output_dir output
```

Check `scripts/gen_chat.sh` and `scripts/gen_preference.sh` for more details.

## Requirements

You can get all the python packages by using `pip install -r requirements.txt`.

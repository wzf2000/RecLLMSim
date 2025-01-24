# Human Vs. Agent: A Comprehensive Analysis of Task-Oriented Conversations with LLMs

## Data

We provide all the data files under the `real_human_user` and `LLM_agent_user` folder.

The conversation dataset is divided into several parts according to the task and the user. Each conversation is stored in a `.json` file under the corresponding task folder.

The format of the `.json` file is:
```json
{
    "task": "The name of the task for this conversation.",
    "preference_id": "The ID of the user profile used in this conversation.",
    "task_context_id": "The ID of the task description used in this conversation.",
    "preference": "Preference used in this conversation.",
    "preference_zh": "Preference used in this conversation translated in Chinese.",
    "task_context": "The scenario-specific task description used in this conversation",
    "task_context_zh": "The scenario-specific task description used in this conversation translated in Chinese.",
    "history": [
        {
            "role": "user",
            "content": "Some text generated by user simulator LLM.",
            "content_zh": "Some text generated by user simulator LLM translated in Chinese.",
            "intent": "The intent annotation of the user simulator LLM.",
            "intent_zh": "The intent annotation of the user simulator LLM translated in Chinese.",
        },
        {
            "role": "assistant",
            "content": "Some text generated by assistant LLM.",
            "content_zh": "Some text generated by assistant LLM translated in Chinese.",
            "hallucination": {
                "hallucination": "Whether the assistant hallucinates in this turn.",
                "memo": "The memo of the hallucination.",
            },
        },
        // ...
    ],
    "conflict": false,
    "preference_summary": "The summary of the user simulator's preference extracted from the conversation.",
    "rating": {
        // Some rating information.
    },
    // Some other information.
}
```

## User Simulator Framework

We provide the code for the framework in the `RecUserSim` folder.

Check the `readme.md` file in the `RecUserSim` folder for more details.

## File Structure

- `real_human_user`: The conversation data constructed from real human users.
- `LLM_agent_user`: The conversation data constructed by the LLM agent users.
- `RecUserSim`: The code for the user simulator framework.
- `README.md`: This file.
- `user_behavior`: Automatically label method for user behavior analysis.
- `user_profiling`: The code for our profile identification experiments.
- `statistics`: The code for the statistics of the dataset.

## Some Statistics

### Simulation Quality across Different Conversation Turns

| Turns | # Conv. | Preference Alignment | Role-Playing Completeness |
| :---: | :-----: | :------------------: | :-----------------------: |
|   1   |    1    |        1.0000        |          1.0000           |
|   2   |   316   |        1.2089        |          1.2911           |
|   3   |   575   |        1.3496        |          1.6417           |
|   4   |   452   |        1.6018        |          1.6416           |
|   5   |   279   |        1.7097        |          1.6093           |
|   6   |   138   |        1.8043        |          1.7246           |
|   7   |   69    |        1.7536        |          1.7101           |
|   8   |   15    |      **1.8667**      |        **1.7333**         |
|   9   |    6    |        1.6667        |          1.3333           |
|  10   |    5    |        1.6000        |          1.2000           |

## Annotation Criteria

### Detail Level

- **2 (High):** In-depth responses with detailed explanations (e.g., a recipe including specific ingredients and step-by-step instructions).
- **0 (Low):** Vague or brief responses (e.g., listing destination names without descriptions in a travel plan).
- **1 (Medium):** Responses between these extremes.

### Diversity

- **Low (0):** Only one or two options provided per query.
- **Medium (1):** Some queries have three or more options, while others are limited.
- **High (2):** All queries include three or more diverse options.

### Practical Utility

- **Not usable (0):** Unrealistic plans (e.g., an overly tight travel schedule or learning plans with unreasonably fast progress).
- **Somewhat usable (1):** Partially applicable suggestions (e.g., gift ideas that provide inspiration but are difficult to implement).
- **Highly usable (2):** Practical and actionable plans (e.g., a Python learning plan with achievable milestones).

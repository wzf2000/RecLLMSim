import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam, ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel
from enum import Enum

file_name = '../LLM_agent_user/new travel planning/109.json'
# get the first directory in the file path
human = 'LLM_agent_user' not in file_name
format_version = 0 # 1, 2, 3, add more versions if needed

# Function to format the conversation history for the prompt
def conv_format(history: list[dict[str, str]], version: int) -> str:
    if version == 0:
        # original format
        text = ''
        for turn in history:
            if turn['role'] == 'user':
                text += f'User: {turn["content"]}\n\n'
            else:
                text += f'System: {turn["content"]}\n\n'
        return text
    elif version == 1:
        # raw json format
        return json.dumps(history, ensure_ascii=False, indent=4)
    elif version == 2:
        # filtered json format
        ret = []
        for utt in history:
            ret.append({
                'role': utt['role'],
                'content': utt['content'],
                'content_zh': utt.get('content_zh', ''),
            })
        return json.dumps(ret, ensure_ascii=False, indent=4)
    elif version == 3:
        # original format with Chinese content
        text = ''
        for turn in history:
            if turn['role'] == 'user':
                # text += f'User: {turn["content"]}\n\n'
                if 'content_zh' in turn:
                    text += f'User (Chinese): {turn["content_zh"]}\n\n'
            else:
                # text += f'System: {turn["content"]}\n\n'
                if 'content_zh' in turn:
                    text += f'System (Chinese): {turn["content_zh"]}\n\n'
        return text
    else:
        raise ValueError(f'Unsupported version: {version}')

prompt_template = """You are a skilled conversational analyst. Your task is to evaluate a given dialogue in [Language] between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

[Insert Aspects Here]

**Instructions:**

1. Review the entire dialogue between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]

**Output Format:**

The output should be in JSON format, structured like this:
```json
[Insert JSON Format Here]
```

**Your Response:**"""

aspect_prompt = """### Interaction Style

1. **Context Dependency** (How much the user's questions depend on prior context or information in the conversation):
    - **1 - Very Independent:**
      The user's questions are mostly self-contained and do not rely on prior context.
      *Example:* "What is the best way to make pasta?"

    - **2 - Independent:**
      The user's questions have minimal reliance on prior context, with only occasional connections to previous questions.
      *Example:* "What’s a good gift for a teenager?" followed by "How about for someone who loves art?"

    - **3 - Moderately Dependent:**
      The user's questions partially build on prior context, showing some reliance on earlier parts of the conversation.
      *Example:* "What are some tips for learning guitar chords?" followed by "Can you recommend exercises for switching between them quickly?"

    - **4 - Dependent:**
      The user's questions strongly depend on prior context and directly reference earlier discussions.
      *Example:* "What are the best walking routes in Rome?" followed by "Are there any good restaurants along those routes?"

    - **5 - Very Dependent:**
      The user's questions are completely tied to prior context, making them difficult to understand without the preceding conversation.
      *Example:* "Which one would you recommend?" following a detailed discussion about several specific gift ideas.

2. **Explanation:** Assess how often the user provides reasons for their own requests or preferences during the conversation. Noticed that request itself (anything you need to response) or stating a preference should not be considered as a reason. A "reason" should only be count when the request or the preference is in the same turn with its reason.
      *No Explanation Example:* "I would prefer public transportation."
      *Explanation Example:* "I would prefer public transportation as it would allow me to immerse myself more in the local culture."
    - **Frequent Explanation:** The user separately explicitly explains their reasoning in **two or more** turns during the entire conversation. These reasons must refer to different aspects of the conversation and are clearly separated in two or more different turns, not nested in one long turn.
    - **Rare Explanation:** The user gives a reason in **only one** turn during the entire conversation.
    - **No Explanation:** The user does not provide any reason or justification for their requests or preferences in any turn.
    
    You should give the corresponding reason for the user's explanation in the `reason_for_explanation` field in the output.

3. **Promise:** Determine whether the user explicitly commits to following the assistant’s suggestion after receiving it (not in their initial request).
    - **Have Promise:** The user explicitly confirms or imply he/she will follow the assistant’s suggestion. This includes expressions of future intent, agreement to act, or direct statements of commitment. 
        *Eample:* "I will book the group tours in advance."
        *Helpful Keywords (Users' promises may include the following keywords.):* "I will", "I’ll", "I will definitely", "I will certainly", "I will follow your suggestion", "I will take your advice", "I will try that", "I will do that", "I will go with that".
    - **No Promise:** The user does not confirm or imply they will follow the assistant’s suggestion. 
        *Notice:* If the user only responds to the assistant's question, it should not be considered a promise.
        *Example:*
            System: "It will be helpful if you tell me your budget."
            User: "I have a budget of 1000 dollars."
    
    You should give the corresponding reason for the user's promise in the `reason_for_promise` field in the output.

4. **Feedback Attitude**:
    - **No Feedback**: Throughout the multi-turn dialogue, the user does not provide any feedback or reaction to the responses received, proceeding with their inquiries without acknowledging the answers.
    - **Positive Feedback**: The user consistently provides affirmative or approving responses to the information received throughout the interaction, indicating satisfaction or agreement.
    - **Negative Feedback**: The user consistently expresses dissatisfaction or disagreement with the responses during the interaction, possibly indicating the information was unhelpful or incorrect.
    - **Both Feedback**: The user provides both positive and negative feedback at different times throughout the dialogue, showing varied reactions depending on the responses.

5. **Politeness**
    - **Polite**: The user's language style is always very polite.
        *Example:* "Thank you very much for the detailed instructions on booking the shuttle.“
    - **Neutral**: The user's language style is neutral, neither overly polite nor rude. The normal tone is considered neutral.
    - **Impolite**: The user has not promised to take action based on the recommendation by the assistant.
        *Example:* "This is too damn expensive."

6. **Oral or Formal**
    - **Oral**: The user's language style is always oral.
    - **Formal**: The user's language style is always formal."""

format_prompt = """{
    "context_dependency": "1-5",
    "explanation": "Frequent/Rare/No",
    "reason_for_explanation": "A string explaining the reason for the user's explanation, if any.",
    "promise": "Have/no",
    "reason_for_promise": "A string explaining the reason for the user's promise, if any.",
    "feedback": "NoFeedback/Positive/Negative/Both"
    "politeness": "Polite/Neutral/Impolite",
    "formality": "Oral/Formal"
}"""

class Rating(int, Enum):
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5

class Explanation(str, Enum):
    Frequent = 'Frequent'
    Rare = 'Rare'
    NoExplanation = 'NoExplanation'

class Promise(str, Enum):
    HavePromise = 'HavePromise'
    NoPromise = 'NoPromise'

class Feedback(str, Enum):
    NoFeedback = 'NoFeedback'
    Positive = 'Positive'
    Negative = 'Negative'
    Both = 'Both'

class Politeness(str, Enum):
    Polite = 'Polite'
    Neutral = 'Neutral'
    Impolite = 'Impolite'

class Formality(str, Enum):
    Oral = 'Oral'
    Formal = 'Formal'

class Strategy(BaseModel):
    context_dependency: Rating
    explanation: Explanation
    reason_for_explanation: str
    promise: Promise
    reason_for_promise: str
    feedback: Feedback
    politeness: Politeness
    formality: Formality

client = OpenAI(
    base_url="https://svip.xty.app/v1",
    api_key="sk-7V69yk27IW2eTvth7543Fc9677Bb4aCc9aD546C723FaFf3d"
)

def get_prompt(history: list[dict[str, str]], human: bool, version: int) -> str:
    prompt = prompt_template
    if human:
        prompt = prompt.replace('[Language]', 'Chinese')
    else:
        prompt = prompt.replace('[Language]', 'English')
    prompt = prompt.replace('[Insert Aspects Here]', aspect_prompt)
    prompt = prompt.replace('[Insert JSON Format Here]', format_prompt)
    prompt = prompt.replace('[Insert Dialogue Here]', conv_format(history, version))
    return prompt

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def cls(history: list[dict[str, str]], model: str, version: int, human: bool):
    input_text = get_prompt(history, human, version)
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content="You are a skilled conversational analyst.",
            role="system"
        ),
#         ChatCompletionSystemMessageParam(
#             content="""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
# Knowledge cutoff: 2023-10
# Current date: 2025-07-09

# Personality: v2""",
#             role="system"
#         ),
        ChatCompletionUserMessageParam(
            content=input_text,
            role="user"
        )
    ]
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=Strategy,
    )
    ret = response.choices[0].message.parsed
    if ret is None:
        print(f"Response parsing failed for model {model} with version {version}.")
        raise ValueError("Response parsing failed.")
    else:
        return ret.model_dump()

if __name__ == "__main__":
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'history' not in data:
        print(f'No history found in {file_name}')
        exit(1)
    history = data['history']
    result = cls(history, 'gpt-4o-2024-08-06', format_version, human)
    print(result)

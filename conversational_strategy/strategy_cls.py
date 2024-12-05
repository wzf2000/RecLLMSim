import os
import json
from output_format import StrategyV1, StrategyV2
from openai import LengthFinishReasonError, OpenAI

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

human_prompt_V1 = """You are a skilled conversational analyst. Your task is to evaluate a given dialogue, which is in Chinese, between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

1. **User Strategy Type**: Determine whether the user's approach is best described as a "Planning-Oriented Inquiry" or a "Sequential Information Request".
   
   - **Planning-Oriented Inquiry**: The user explicitly requests information in a structured manner, often outlining steps, sequences, or processes to achieve a particular goal.

   - **Sequential Information Request**: The user asks for information in a logical order that reflects an underlying plan or roadmap they intend to follow without explicitly outlining it.

2. **Depth vs. Breadth Strategy**:

   - **Depth-Oriented**: The user delves deeply into a single topic before moving on.
   
   - **Breadth-Oriented**: The user explores multiple topics with less depth.
   
   - **Comprehensive Strategy**: The user balances depth and breadth, exploring multiple topics with sufficient depth in each.

3. **Context Dependency**:

   - **High Context Dependency**: The user's questions heavily rely on previous dialogue content.
   
   - **Low Context Dependency**: The user's questions can be understood independently of earlier dialogue.

4. **Question Specificity**:

   - **Broad**: The user's questions are open-ended and cover a wide range of topics.
   
   - **Specific**: The user's questions are specific and focused on detailed information.

**Instructions:**

1. Review the entire dialogue, which is provided in Chinese, between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]

**Output Format:**

The output should be in JSON format, structured like this:
```json
{
    "planning": "Planning/Sequential",
    "order": "Depth/Breadth/Comprehensive",
    "context": "High/Low",
    "question": "Broad/Specific"
}
```

**Your Response:**
"""

human_prompt_V2 = """You are a skilled conversational analyst. Your task is to evaluate a given dialogue, which is in Chinese, between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

### 1. Depth vs. Breadth Strategy

- **Depth-Oriented**: The user focuses intensely on a specific aspect of their plan, thoroughly exploring it before moving on to another aspect.

- **Breadth-Oriented**: The user considers a wide range of aspects of their plan, exploring each one briefly without going into much detail.

- **Depth-First, Then Breadth**: The user initially examines a particular aspect of their plan in great detail before expanding their inquiry to include a broader range of aspects.

- **Breadth-First, Then Depth**: The user begins by covering a wide array of aspects superficially and subsequently chooses specific aspects to explore in detail.

### 2. User Feedback to Responses

- **No Feedback**: Throughout the multi-turn dialogue, the user does not provide any feedback or reaction to the responses received, proceeding with their inquiries without acknowledging the answers.

- **Positive Feedback**: The user consistently provides affirmative or approving responses to the information received throughout the interaction, indicating satisfaction or agreement.

- **Negative Feedback**: The user consistently expresses dissatisfaction or disagreement with the responses during the interaction, possibly indicating the information was unhelpful or incorrect.

- **Both Feedback**: The user provides both positive and negative feedback at different times throughout the dialogue, showing varied reactions depending on the responses.

**Instructions:**

1. Review the entire dialogue, which is provided in Chinese, between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]

**Output Format:**

The output should be in JSON format, structured like this:
```json
{
    "order": "Depth/Breadth/DepthBreadth/BreadthDepth",
    "feedback": "NoFeedback/Positive/Negative/Both"
}
```

**Your Response:**
"""

sim_prompt_V1 = """You are a skilled conversational analyst. Your task is to evaluate a given dialogue between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

1. **User Strategy Type**: Determine whether the user's approach is best described as a "Planning-Oriented Inquiry" or a "Sequential Information Request".
   
   - **Planning-Oriented Inquiry**: The user explicitly requests information in a structured manner, often outlining steps, sequences, or processes to achieve a particular goal.

   - **Sequential Information Request**: The user asks for information in a logical order that reflects an underlying plan or roadmap they intend to follow without explicitly outlining it.

2. **Depth vs. Breadth Strategy**:

   - **Depth-Oriented**: The user delves deeply into a single topic before moving on.
   
   - **Breadth-Oriented**: The user explores multiple topics with less depth.
   
   - **Comprehensive Strategy**: The user balances depth and breadth, exploring multiple topics with sufficient depth in each.

3. **Context Dependency**:

   - **High Context Dependency**: The user's questions heavily rely on previous dialogue content.
   
   - **Low Context Dependency**: The user's questions can be understood independently of earlier dialogue.

4. **Question Specificity**:

   - **Broad**: The user's questions are open-ended and cover a wide range of topics.
   
   - **Specific**: The user's questions are specific and focused on detailed information.

**Instructions:**

1. Review the entire dialogue between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]

**Output Format:**

The output should be in JSON format, structured like this:
```json
{
    "planning": "Planning/Sequential",
    "order": "Depth/Breadth/Comprehensive",
    "context": "High/Low",
    "question": "Broad/Specific"
}
```

**Your Response:**
"""

sim_prompt_V2 = """You are a skilled conversational analyst. Your task is to evaluate a given dialogue between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

### 1. Depth vs. Breadth Strategy

- **Depth-Oriented**: The user focuses intensely on a specific aspect of their plan, thoroughly exploring it before moving on to another aspect.

- **Breadth-Oriented**: The user considers a wide range of aspects of their plan, exploring each one briefly without going into much detail.

- **Depth-First, Then Breadth**: The user initially examines a particular aspect of their plan in great detail before expanding their inquiry to include a broader range of aspects.

- **Breadth-First, Then Depth**: The user begins by covering a wide array of aspects superficially and subsequently chooses specific aspects to explore in detail.

### 2. User Feedback to Responses

- **No Feedback**: Throughout the multi-turn dialogue, the user does not provide any feedback or reaction to the responses received, proceeding with their inquiries without acknowledging the answers.

- **Positive Feedback**: The user consistently provides affirmative or approving responses to the information received throughout the interaction, indicating satisfaction or agreement.

- **Negative Feedback**: The user consistently expresses dissatisfaction or disagreement with the responses during the interaction, possibly indicating the information was unhelpful or incorrect.

- **Both Feedback**: The user provides both positive and negative feedback at different times throughout the dialogue, showing varied reactions depending on the responses.

**Instructions:**

1. Review the entire dialogue between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]

**Output Format:**

The output should be in JSON format, structured like this:
```json
{
    "order": "Depth/Breadth/DepthBreadth/BreadthDepth",
    "feedback": "NoFeedback/Positive/Negative/Both"
}
```

**Your Response:**
"""

def conv_format(history: list[dict[str, str]]) -> str:
    text = ''
    for turn in history:
        if turn['role'] == 'user':
            text += f'User: {turn["content"]}\n\n'
        else:
            text += f'System: {turn["content"]}\n\n'
    return text

def cls(text: str, model: str, version: int, human: bool) -> StrategyV1 | StrategyV2:
    if human:
        if version == 1:
            prompt = human_prompt_V1
        elif version == 2:
            prompt = human_prompt_V2
        else:
            assert False
    else:
        if version == 1:
            prompt = sim_prompt_V1
        elif version == 2:
            prompt = sim_prompt_V2
        else:
            assert False
    input_text = prompt.replace('[Insert Dialogue Here]', text)
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled conversational analyst."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0.0,
            response_format=StrategyV1 if version == 1 else StrategyV2,
        ).choices[0].message
        if response.parsed:
            return response.parsed
        elif response.refusal:
            print("Refusal!")
            return cls(text)
    except LengthFinishReasonError as e:
        print(f"Too many tokens: {e}")
    except Exception as e:
        print(f"Error: {e}")
    return cls(text)

def multi_cls_V1(text: str, model: str, human: bool) -> dict[str, dict[str, str]]:
    strategy1 = cls(text, model, 1, human)
    strategy2 = cls(text, model, 1, human)
    strategy3 = cls(text, model, 1, human)
    # get the most common strategy
    if strategy1.information_request == strategy2.information_request or strategy1.information_request == strategy3.information_request:
        planning = strategy1.information_request
    elif strategy2.information_request == strategy3.information_request:
        planning = strategy2.information_request
    else:
        assert False
    if strategy1.order == strategy2.order or strategy1.order == strategy3.order:
        order = strategy1.order
    elif strategy2.order == strategy3.order:
        order = strategy2.order
    else:
        assert False
    if strategy1.context == strategy2.context or strategy1.context == strategy3.context:
        context = strategy1.context
    elif strategy2.context == strategy3.context:
        context = strategy2.context
    else:
        assert False
    if strategy1.question == strategy2.question or strategy1.question == strategy3.question:
        question = strategy1.question
    elif strategy2.question == strategy3.question:
        question = strategy2.question
    else:
        assert False
    final_strategy = StrategyV1(information_request=planning, order=order, context=context, question=question)
    return {
        'final': final_strategy.model_dump(),
        '1': strategy1.model_dump(),
        '2': strategy2.model_dump(),
        '3': strategy3.model_dump()
    }

def multi_cls_V2(text: str, model: str, human: bool) -> dict[str, dict[str, str]]:
    strategy1 = cls(text, model, 2, human)
    strategy2 = cls(text, model, 2, human)
    strategy3 = cls(text, model, 2, human)
    # get the most common strategy
    if strategy1.order == strategy2.order or strategy1.order == strategy3.order:
        order = strategy1.order
    elif strategy2.order == strategy3.order:
        order = strategy2.order
    else:
        assert False
    if strategy1.feedback == strategy2.feedback or strategy1.feedback == strategy3.feedback:
        feedback = strategy1.feedback
    elif strategy2.feedback == strategy3.feedback:
        feedback = strategy2.feedback
    else:
        assert False
    final_strategy = StrategyV2(order=order, feedback=feedback)
    return {
        'final': final_strategy.model_dump(),
        '1': strategy1.model_dump(),
        '2': strategy2.model_dump(),
        '3': strategy3.model_dump()
    }

def multi_cls(text: str, model: str, version: int, human: bool) -> dict[str, dict[str, str]]:
    if version == 1:
        return multi_cls_V1(text, model, human)
    elif version == 2:
        return multi_cls_V2(text, model, human)
    else:
        assert False

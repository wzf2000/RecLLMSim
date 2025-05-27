import os
import json
from openai import LengthFinishReasonError, OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from output_format import StrategyV1, StrategyV2, StrategyV3

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

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

human_prompt_template = prompt_template.replace('[Language]', 'Chinese')

sim_prompt_template = prompt_template.replace('[Language]', 'English')

aspect_list = ["""1. **User Strategy Type**: Determine whether the user's approach is best described as a "Planning-Oriented Inquiry" or a "Sequential Information Request".
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
   - **Specific**: The user's questions are specific and focused on detailed information.""", """### 1. Depth vs. Breadth Strategy

- **Depth-Oriented**: The user focuses intensely on a specific aspect of their plan, thoroughly exploring it before moving on to another aspect.
- **Breadth-Oriented**: The user considers a wide range of aspects of their plan, exploring each one briefly without going into much detail.
- **Depth-First, Then Breadth**: The user initially examines a particular aspect of their plan in great detail before expanding their inquiry to include a broader range of aspects.
- **Breadth-First, Then Depth**: The user begins by covering a wide array of aspects superficially and subsequently chooses specific aspects to explore in detail.

### 2. User Feedback to Responses

- **No Feedback**: Throughout the multi-turn dialogue, the user does not provide any feedback or reaction to the responses received, proceeding with their inquiries without acknowledging the answers.
- **Positive Feedback**: The user consistently provides affirmative or approving responses to the information received throughout the interaction, indicating satisfaction or agreement.
- **Negative Feedback**: The user consistently expresses dissatisfaction or disagreement with the responses during the interaction, possibly indicating the information was unhelpful or incorrect.
- **Both Feedback**: The user provides both positive and negative feedback at different times throughout the dialogue, showing varied reactions depending on the responses.""", """### 1. **Question Broadness** (How specific or broad the user's questions are overall in the conversation):
- **1 - Very Broad:**
  The user's questions are extremely general or vague, with little to no specific focus.
  *Example:* "What are some fun places to visit?"

- **2 - Broad:**
  The user's questions show some focus but still remain relatively general, allowing for various interpretations.
  *Example:* "What are the best tourist destinations in Europe?"

- **3 - Moderate:**
  The user's questions are a mix of general and specific, showing moderate specificity but also including broader inquiries.
  *Example:* "What are the must-see landmarks in Paris during spring?"

- **4 - Specific:**
  The user's questions are primarily focused and detailed, narrowing down the scope to particular areas or topics.
  *Example:* "What are some good family-friendly activities in Kyoto during cherry blossom season?"

- **5 - Very Specific:**
  The user's questions are highly detailed and precise, consistently asking well-scoped and focused questions.
  *Example:* "How can I plan a one-day itinerary in Kyoto to visit Kinkaku-ji, Fushimi Inari, and Gion, including transportation options?"

### 2. **Context Dependency** (How much the user's questions depend on prior context or information in the conversation):
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

### 3. **User Feedback Analysis** (How the user responds to the AI's answers):
- **Positive:**
  The user explicitly provides positive feedback about the AI's responses, such as expressing satisfaction, appreciation, or agreement.
  *Example:* "That’s very helpful, thank you!" or "Great explanation!"

- **Negative:**
  The user explicitly provides negative feedback, such as expressing dissatisfaction, disagreement, or pointing out issues with the AI's responses.
  *Example:* "This isn’t what I was asking for," or "That doesn’t make sense."

- **NoFeedback:**
  The user does not provide any explicit feedback, neither positive nor negative, and the conversation continues without evaluative comments.
  *Example:* The user only asks questions or makes neutral statements without commenting on the quality of answers.

- **Both:**
  Across multiple turns in the conversation, the user provides both explicit positive and explicit negative feedback in different parts of the interaction.
  *Example:* In one turn, the user says, "Thanks, that’s really helpful!" (Positive), and in another turn, they say, "Actually, that’s not what I meant" (Negative)."""]

format_list = ["""{
    "planning": "Planning/Sequential",
    "order": "Depth/Breadth/Comprehensive",
    "context": "High/Low",
    "question": "Broad/Specific"
}""", """{
    "order": "Depth/Breadth/DepthBreadth/BreadthDepth",
    "feedback": "NoFeedback/Positive/Negative/Both"
}""", """{
    "question_broadness": "1-5",
    "context_dependency": "1-5",
    "feedback": "NoFeedback/Positive/Negative/Both"
}"""]

return_type_list = [StrategyV1, StrategyV2, StrategyV3]

def conv_format(history: list[dict[str, str]]) -> str:
    text = ''
    for turn in history:
        if turn['role'] == 'user':
            text += f'User: {turn["content"]}\n\n'
        else:
            text += f'System: {turn["content"]}\n\n'
    return text

def get_prompt(text: str, human: bool, version: int) -> str:
    prompt = prompt_template
    if human:
        prompt = prompt.replace('[Language]', 'Chinese')
    else:
        prompt = prompt.replace('[Language]', 'English')
    prompt = prompt.replace('[Insert Aspects Here]', aspect_list[version - 1])
    prompt = prompt.replace('[Insert JSON Format Here]', format_list[version - 1])
    prompt = prompt.replace('[Insert Dialogue Here]', text)
    return prompt

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def cls(text: str, model: str, version: int, human: bool, n: int = 3) -> list[StrategyV1 | StrategyV2 | StrategyV3]:
    input_text = get_prompt(text, human, version)
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
        response_format=return_type_list[version - 1],
        n=n,
    )
    for choice in response.choices:
        if choice.finish_reason == LengthFinishReasonError:
            print(f"Length finish reason error: {choice}")
            raise ValueError("Length finish reason error.")
        if choice.message.parsed is None:
            print(f"Response parsing failed: {choice}")
            raise ValueError("Response parsing failed.")
    ret = [choice.message.parsed for choice in response.choices]
    if len(ret) != n:
        print(f"Response length mismatch: {len(ret)} != {n}")
        raise ValueError("Response length mismatch.")
    return ret

def ensemble_answer(answer1: str | int, answer2: str | int, answer3: str | int) -> str | int:
    if isinstance(answer1, str):
        if answer1 == answer2 or answer1 == answer3:
            return answer1
        elif answer2 == answer3:
            return answer2
        else:
            print(f"Disagreement: {answer1}, {answer2}, {answer3}")
            assert False, f"{answer1}, {answer2}, {answer3}"
    else:
        # If max - min <= 2, return the average
        if max(answer1, answer2, answer3) - min(answer1, answer2, answer3) <= 2:
            return (answer1 + answer2 + answer3) // 3 + (1 if (answer1 + answer2 + answer3) % 3 == 2 else 0)
        else:
            print(f"Disagreement: {answer1}, {answer2}, {answer3}")
            assert False, f"{answer1}, {answer2}, {answer3}"

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def multi_cls(text: str, model: str, version: int, human: bool) -> dict[str, dict[str, str | int]]:
    strategies = cls(text, model, version, human, n=3)
    strategies = [strategy.model_dump() for strategy in strategies]
    keys = strategies[0].keys()
    final_strategy = {}
    for key in keys:
        final_strategy[key] = ensemble_answer(strategies[0][key], strategies[1][key], strategies[2][key])
    return {
        'final': final_strategy,
        '1': strategies[0],
        '2': strategies[1],
        '3': strategies[2]
    }

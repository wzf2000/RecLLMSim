import os
import json
import time
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from tenacity import retry, stop_after_attempt, wait_fixed

from output_format import strategy_list, template_ids, StrategyType

api_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_config.json')

with open(api_config_file, 'r') as f:
    api_config = json.load(f)

client = OpenAI(
    base_url=api_config['base_url'],
    api_key=api_config['api_key']
)

prompt_templates = ["""You are a skilled conversational analyst. Your task is to evaluate a given dialogue in [Language] between a user and a system to identify several aspects of the user's inquiry strategy. Specifically, you need to classify each aspect as follows:

[Insert Aspects Here]

**Instructions:**

1. Review the entire dialogue between the user and the system.
2. For each aspect, identify key indicators based on the user's questions and requests.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]""", """You are a skilled conversational analyst. Your task is to evaluate a given dialogue between a user and a system (assistant) to identify several aspects of assistant’s answers. Specifically, you need to classify each aspect as follows:

[Insert Aspects Here]

**Instructions:**

1. Review the entire dialogue between the user and the assistant.
2. For each aspect, identify key indicators based on the assistant’s answers.
3. Provide classifications for each aspect and justify your decisions with examples from the dialogue.

**Dialogue:**

[Insert Dialogue Here]"""]

aspect_list = ["""1. **Problem Solving Approach Type**:
    - **All-in-One**: Users adopting this strategy explicitly request a comprehensive plan at the outset, seeking a structured solution framework before addressing finer details. This top-down approach reflects a preference for upfront clarity.
        - **Example**: 
            - 1st turn: "I'm in the process of planning a two-day trip this winter, and I would appreciate your help in refining my travel plans. I need assistance with accommodations, transportation, and creating a comprehensive itinerary. Can you help me with that?"
    - **Step-by-Step**: Users following this strategy iteratively refine their solutions through incremental exchanges, dynamically adjusting their requests without articulating an overarching plan initially.
        - **Example**: 
            - 1st turn: "I'm thinking of a destination that offers a mix of art, unique dining experiences, and a bit of adventure. I’m based in New York City. Can you suggest a few destinations that fit this criteria?"
            - 2nd turn: "It would be convenient if I could walk to some galleries or have easy access to public transportation. As for the stay, I'd prefer a boutique hotel with a touch of local art in its décor. Can you help me with that?"

2. **Order of Multi-Aspect Optimization**:
    - Different aspects including but not limited to:
        - Travel planning: destination, accommodation, transportation
        - Preparing gifts: gift option, price, platform
        - Recipe planning: recipe, ingredients, cooking method/order
        - Skill learning planning: time planning, relevant information, skill choices
    - Category into four types of order:
        - **Depth-Oriented**: The user focuses intensely on a specific aspect of their plan, thoroughly exploring it before moving on to another aspect.
        - **Breadth-Oriented**: The user considers a wide range of aspects of their plan, exploring each one briefly without going into much detail.
        - **Depth-First, Then Breadth**: The user initially examines a particular aspect of their plan in great detail before expanding their inquiry to include a broader range of aspects.
        - **Breadth-First, Then Depth**: The user begins by covering a wide array of aspects superficially and subsequently chooses specific aspects to explore in detail.""", """### 1. **Question Broadness** (How specific or broad the user's questions are overall in the conversation):
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
  *Example:* "Which one would you recommend?" following a detailed discussion about several specific gift ideas.""", """### Interaction Style
  
1. **Explanation:** Assess how often the user provides reasons for their own requests or preferences during the conversation. Noticed that request itself (anything you need to response) or stating a preference should not be considered as a reason. A "reason" should only be count when the request or the preference is in the same turn with its reason.
      *No Explanation Example:* "I would prefer public transportation."
      *Explanation Example:* "I would prefer public transportation as it would allow me to immerse myself more in the local culture."
    - **Frequent Explanation:** The user separately explicitly explains their reasons **two or more** times during the entire conversation. These reasons must refer to different aspects of the conversation and are clearly separated in two or more different turns, not nested in one long turn.
    - **Rare Explanation:** The user gives a reason in **only one** time during the entire conversation.
    - **No Explanation:** The user does not provide any reason or justification for their requests or preferences in any turn.

    You should give the corresponding reason for the user's explanation in the `reason_for_explanation` field in the output.

2. **Promise:** Determine whether the user explicitly commits to following the assistant’s suggestion after receiving it (not in their initial request).
    - **Have Promise:** The user explicitly confirms or imply he/she will follow the assistant’s suggestion. This includes expressions of future intent, agreement to act, or direct statements of commitment.
        *Eample:* "I will book the group tours in advance."
        *Helpful Keywords (Users' promises may include the following keywords.):* "I will", "I’ll", "I will definitely", "I will certainly", "I will follow your suggestion", "I will take your advice", "I will try that", "I will do that", "I will go with that".
    - **No Promise:** The user does not confirm or imply they will follow the assistant’s suggestion.
        *Notice:* If the user only responds to the assistant's question, it should not be considered a promise.
        *Example:*
            System: "It will be helpful if you tell me your budget."
            User: "I have a budget of 1000 dollars."

    You should give the corresponding reason for the user's promise in the `reason_for_promise` field in the output.

3. **Feedback Attitude**:
    - **No Feedback**: Throughout the multi-turn dialogue, the user does not provide any feedback or reaction to the responses received, proceeding with their inquiries without acknowledging the answers.
    - **Positive Feedback**: The user consistently provides affirmative or approving responses to the information received throughout the interaction, indicating satisfaction or agreement.
    - **Negative Feedback**: The user consistently expresses dissatisfaction or disagreement with the responses during the interaction, possibly indicating the information was unhelpful or incorrect.
    - **Both Feedback**: The user provides both positive and negative feedback at different times throughout the dialogue, showing varied reactions depending on the responses.

4. **Politeness**
    - **Polite**: The user's language style is always very polite.
        *Example:* "Thank you very much for the detailed instructions on booking the shuttle.“
    - **Neutral**: The user's language style is neutral, neither overly polite nor rude. The normal tone is considered neutral.
    - **Impolite**: The user has not promised to take action based on the recommendation by the assistant.
        *Example:* "This is too damn expensive."

5. **Formality**
    - **Oral**: The user's language style is always oral.
    - **Formal**: The user's language style is always formal.
    - **Mixed**: The user's language style is both oral and formal.""", """### Conversation Evaluation
1. **Utility**: Measures whether the assistant's suggestion aligns with the user's intent and provides substantive value toward solving their task.
    - **High Utility**: Assistant directly addresses the user’s request with relevant and helpful suggestions.
      *Example:* Specific restaurant recommendations that match stated preferences.
    - **Moderate Utility**: Assistant's answers are somewhat helpful, but contains irrelevant, generic, or partially mismatched information.
      *Example:* Broad tourist ideas without user constraints considered.
    - **Low Utility**: Assistant fails to address the user’s need or goes off-topic. 
      *Example:* Recommends travel gear when user asked about travel plan.

2. **Operability**: Evaluates whether the assistant’s recommendation is concrete, executable, and feasible in a real-world setting.
    - **High Operability**: Assistant provides clear next steps, specific actions, or well-defined options.
      *Example:* “Book a 7 PM sushi class at Tsukiji Market.”
    - **Moderate Operability**: Assistant provides general advice, lacks specifics but can still guide user action.
      *Example:* “Try searching for cooking classes in Tokyo.”
    - **Low Operability**: Assistant's answers are too vague, aspirational, or impractical to implement.
      *Example:* “Immerse yourself in local culture through exploration.”
"""]

format_list = ["""{
    "problem_solving": "AllInOne/StepByStep",
    "order": "Depth/Breadth/DepthBreadth/BreadthDepth",
}""", """{
    "question_broadness": "1-5",
    "context_dependency": "1-5",
}""", """{
    "explanation": "Frequent/Rare/No",
    "promise": "Have/no",
    "feedback": "NoFeedback/Positive/Negative/Both"
    "politeness": "Polite/Neutral/Impolite",
    "formality": "Oral/Formal/Mixed"
}""", """{
    "utility": "High/Moderate/Low",
    "operability": "High/Moderate/Low"
}"""]

def conv_format(history: list[dict[str, str]]) -> str:
    text = ''
    for turn in history:
        if turn['role'] == 'user':
            text += f'User: {turn["content"]}\n\n'
        else:
            text += f'System: {turn["content"]}\n\n'
    return text

def get_prompt(text: str, human: bool, version: int) -> str:
    prompt = prompt_templates[template_ids[version - 1]]
    if human:
        prompt = prompt.replace('[Language]', 'Chinese')
    else:
        prompt = prompt.replace('[Language]', 'English')
    prompt = prompt.replace('[Insert Aspects Here]', aspect_list[version - 1])
    prompt = prompt.replace('[Insert Dialogue Here]', text)
    return prompt

def get_messages(prompt: str) -> list[ChatCompletionMessageParam]:
    # get today's date in YYYY-MM-DD format
    today = time.strftime("%Y-%m-%d", time.localtime())
    return [
        ChatCompletionSystemMessageParam(
            content="You are a skilled conversation analyzer. Please just answer the questions and do not respond unrelated things.",
            role="system"
        ),
        ChatCompletionUserMessageParam(
            content=prompt,
            role="user"
        )
    ]

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def parse(model, messages: list[ChatCompletionMessageParam], return_type: type[StrategyType]) -> StrategyType:
    format_response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0,
        top_p=0.9,
        response_format=return_type
    ).choices[0].message.parsed
    if format_response is None:
        print(f"Response parsing failed for model {model}.")
        raise ValueError("Response parsing failed.")
    return format_response

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def cls(text: str, model: str, version: int, human: bool, n: int = 3) -> list[StrategyType]:
    input_text = get_prompt(text, human, version)
    messages = get_messages(input_text)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=0.9,
        n=n,
    )
    ret_list: list[StrategyType] = []
    for choice in response.choices:
        ret = choice.message.content
        if ret == "I'm sorry, I can't assist with that.":
            print(f"The model {model} with version {version} refused to assist with the request.")
            raise ValueError("The model refused to assist with the request.")
        if ret is None:
            print(f"Response parsing failed for model {model} with version {version}.")
            raise ValueError("Response parsing failed.")
        format_messages = get_messages(f"Please format the following response into JSON format:\n{ret}\nThe response should be in the format:\n{format_list[version - 1]}")
        ret = parse(model, format_messages, strategy_list[version - 1])
        ret_list.append(ret)
    assert len(ret_list) == n, "Some strategies are None or response length mismatch."
    return ret_list

def ensemble_answer(answer1: str | int, answer2: str | int, answer3: str | int) -> str | int:
    if isinstance(answer1, str) and isinstance(answer2, str) and isinstance(answer3, str):
        if answer1 == answer2 or answer1 == answer3:
            return answer1
        elif answer2 == answer3:
            return answer2
        else:
            print(f"Disagreement: {answer1}, {answer2}, {answer3}")
            assert False, f"{answer1}, {answer2}, {answer3}"
    else:
        assert isinstance(answer1, int) and isinstance(answer2, int) and isinstance(answer3, int), \
			f"Answers must be all strings or all integers, got {type(answer1)}, {type(answer2)}, {type(answer3)}"
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

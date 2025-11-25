import os
import json
from loguru import logger
from openai import OpenAI
from typing import Callable
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from usim import init, CONFIG_FILE

def init_config():
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    if 'api_base' in config:
        os.environ["OPENAI_API_BASE"] = config['api_base']
    os.environ["OPENAI_API_KEY"] = config['api_key']
    return config

CONFIG = init_config()

class Chatbot:
    def __init__(self, model: str, temperature: float | None = 0.7, **kwargs) -> None:
        self.model: str = model
        if temperature is None:
            temperature = 0.7
        self.temperature: float = temperature
        self.history = []
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def add_to_history(self, message: str, role: str = 'user') -> None:
        self.history.append({
            'role': role,
            'content': message
        })

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def ask_once(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        assert content is not None, "Response content is None"
        return content

    def ask(self, prompt: str, role: str = 'user') -> str:
        self.add_to_history(prompt, role)
        try:
            response = self.ask_once()
        except RetryError:
            logger.error("Failed to get response from model after several attempts.")
            logger.error(f"Current history: {self.history}")
            raise Exception("Model response failed")
        self.add_to_history(response, 'assistant')
        return response

def chat(chatbot: Chatbot):
    logger.info("输入内容即可进行对话，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        response = chatbot.ask(
            prompt=query,
        )
        logger.info(f"{chatbot.model}: {response}")

def get_chatbot(config: dict = CONFIG, model: str = None, temperature: float = None, **kwargs):
    if temperature is not None:
        config['temperature'] = temperature
    if model is not None:
        config['model'] = model
    chatbot = Chatbot(model=config['model'], temperature=config['temperature'], **kwargs)
    return chatbot

def single_turn_chat(chatbot: Chatbot, chat_id: int, query: str, output: bool = True):
    response = chatbot.ask(
        prompt=query,
    )
    if output:
        logger.debug(f"ChatBot{chat_id} ({chatbot.model}): {response}")
    return response

def default_process(x: str, turn: int) -> str:
    return x

def multi_chat(prompt1_list: str, prompt2_list: str, ending: Callable[[str], bool], process1: Callable[[str, int], str] = None, process2: Callable[[str, int], str] = None, max_turn: int = 10, model1: str = None, model2: str = None) -> tuple[Chatbot, Chatbot, bool]:
    if process1 is None:
        process1 = default_process
    if process2 is None:
        process2 = default_process
    config = init_config()
    if model1 is not None:
        chatbot1 = get_chatbot(config, model1)
    else:
        chatbot1 = get_chatbot(config)
    if model2 is not None:
        chatbot2 = get_chatbot(config, model2)
    else:
        chatbot2 = get_chatbot(config)
    logger.debug('Stage: Prompt Before Start')
    for prompt1 in prompt1_list:
        logger.debug(f'Prompt for ChatBot1 ({chatbot1.model}): {prompt1}')
        response1 = single_turn_chat(chatbot1, 1, prompt1, output=True)
    for prompt2 in prompt2_list:
        logger.debug(f'Prompt for ChatBot2 ({chatbot2.model}): {prompt2}')
        response2 = single_turn_chat(chatbot2, 2, prompt2, output=True)
    logger.debug('Stage: Conversation Start')
    logger.debug(f"ChatBot2 ({chatbot2.model}): {response2}")
    for turn in range(max_turn):
        logger.debug('Turn: {}'.format(turn + 1))
        response1 = single_turn_chat(chatbot1, 1, process1(response2, turn))
        response2 = single_turn_chat(chatbot2, 2, process2(response1, turn))
        # if the user wants to end the conversation(contains the ending string), then end the conversation
        if ending(response2):
            logger.debug('Conversation End by User')
            return chatbot1, chatbot2, True
    return chatbot1, chatbot2, False

if __name__ == '__main__':
    init()
    multi_chat([], ['Any sentences?'], ending=lambda x: 'goodbye' in x.lower(), max_turn=2, model1='gemini-1.5-pro-latest', model2='gemini-1.5-pro-latest')

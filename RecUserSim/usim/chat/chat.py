import os
import json
from loguru import logger
from typing import Callable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

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
    def __init__(self, model: str, temperature: float = 0.5, json_mode: bool = False, **kwargs) -> None:
        if temperature is None:
            temperature = 0.5
        self.temperature: float = temperature
        self.history = []
        if json_mode:
            self.model = ChatOpenAI(
                model=model,
                temperature=self.temperature,
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    }
                },
                streaming=True,
            )
        else:
            self.model = ChatOpenAI(
                model=model,
                temperature=self.temperature,
                streaming=True
            )

    def add_to_history(self, message: str, role: str = 'user') -> None:
        self.history.append({
            'role': role,
            'content': message
        })

    def _history(self):
        return [
            HumanMessage(content=utt['content']) if utt['role'] == 'user' else AIMessage(content=utt['content']) for utt in self.history
        ]

    def ask(self, prompt: str, role='user') -> str:
        self.add_to_history(prompt, role)
        flag = True
        while flag:
            try:
                response = self.model.invoke(input=self._history())
                flag = False
            except Exception as e:
                logger.error(f"Error: {e}")
                flag = True
        self.add_to_history(response.content, 'assistant')
        return response.content

def chat(chatbot: Chatbot):
    logger.info("输入内容即可进行对话，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        response = chatbot.ask(
            prompt=query,
        )
        logger.info(f"ChatGPT: {response}")

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
        logger.info(f"ChatGPT{chat_id}: {response}")
    return response

def default_process(x: str) -> str:
    return x

def multi_chat(prompt1_list: str, prompt2_list: str, ending: Callable[[str], bool], process1: Callable[[str], str] = None, process2: Callable[[str], str] = None, max_turn: int = 10, model1: str = None, model2: str = None):
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
    logger.info('Stage: Prompt Before Start')
    for prompt1 in prompt1_list:
        logger.info(f'Prompt for ChatGPT1: {prompt1}')
        response1 = single_turn_chat(chatbot1, 1, prompt1, output=True)
    for prompt2 in prompt2_list:
        logger.info(f'Prompt for ChatGPT2: {prompt2}')
        response2 = single_turn_chat(chatbot2, 2, prompt2, output=True)
    logger.info('Stage: Conversation Start')
    logger.info(f"ChatGPT2: {response2}")
    for _ in range(max_turn):
        response1 = single_turn_chat(chatbot1, 1, process1(response2))
        response2 = single_turn_chat(chatbot2, 2, process2(response1))
        # if the user wants to end the conversation(contain the ending string), then end the conversation
        if ending(response2):
            logger.success('Conversation End by User')
            break
    return chatbot1, chatbot2

if __name__ == '__main__':
    init()
    multi_chat([], ['Any sentences?'], ending=lambda x: 'goodbye' in x.lower(), max_turn=5)

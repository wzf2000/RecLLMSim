import os
import json
import httpx
import tiktoken
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
    def __init__(self, engine: str, temperature: float = 0.5, max_tokens: int = None, truncate_limit: int = None, json_mode: bool = False, **kwargs) -> None:
        self.engine: str = engine
        if temperature is None:
            temperature = 0.5
        self.temperature: float = temperature
        self.history = []
        self.max_tokens: int = max_tokens or 4000
        self.truncate_limit: int = truncate_limit or (
            126500
            if 'gpt-4o' in engine or 'gpt-4-turbo' in engine or "preview" in engine
            else 30500
            if "gpt-4-32k" in engine
            else 6500
            if "gpt-4" in engine
            else 3500
            if "gpt-3.5-turbo-0613" in engine
            else 14500
        )
        if os.getenv("OPENAI_API_BASE", "https://api.openai.com").startswith('https://svip.xty.app'):
            http_client = httpx.Client(
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com"),
                follow_redirects=True,
            )
            logger.success("Using httpx client for OpenAI API")
        else:
            http_client = None
        if json_mode:
            self.model = ChatOpenAI(
                model=engine,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_kwargs={
                    "response_format": {
                        "type": "json_object"
                    }
                },
                http_client=http_client,
                streaming=True,
            )
        else:
            self.model = ChatOpenAI(
                model=engine,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                http_client=http_client,
                streaming=True
            )

    def add_to_history(self, message: str, role: str = 'user') -> None:
        self.history.append({
            'role': role,
            'content': message
        })

    def get_token_count(self) -> int:
        tiktoken.model.MODEL_TO_ENCODING["gpt-4"] = "cl100k_base"
        encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.history:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 5
            for key, value in message.items():
                if value:
                    num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += 5  # role is always required and always 1 token
        num_tokens += 5  # every reply is primed with <im_start>assistant
        return num_tokens

    def __truncate_conversation(self) -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count() > self.truncate_limit
                and len(self.history) > 1
            ):
                # Don't remove the first message
                self.history.pop(1)
            else:
                break
    
    def _history(self):
        return [
            HumanMessage(content=utt['content']) if utt['role'] == 'user' else AIMessage(content=utt['content']) for utt in self.history
        ]

    def ask(self, prompt: str, role='user') -> str:
        self.add_to_history(prompt, role)
        self.__truncate_conversation()
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

def get_chatbot(config: dict = CONFIG, engine: str = None, temperature: float = None, **kwargs):
    if temperature is not None:
        config['temperature'] = temperature
    if engine is not None:
        config['engine'] = engine
    chatbot = Chatbot(engine=config['engine'], temperature=config['temperature'], **kwargs)
    return chatbot

def single_turn_chat(chatbot: Chatbot, chat_id: int, query: str, output: bool = True):
    response = chatbot.ask(
        prompt=query,
    )
    if output:
        logger.info(f"ChatGPT{chat_id}: {response}")
        # print(f"ChatGPT{chat_id}: ", response, "\n", flush=True)
    return response

def multi_chat(prompt1_list: str, prompt2_list: str, ending: Callable[[str], bool], process1: Callable[[str], str] = None, process2: Callable[[str], str] = None, max_turn: int = 10, engine1: str = None, engine2: str = None):
    if process1 is None:
        process1 = lambda x: x
    if process2 is None:
        process2 = lambda x: x
    config = init_config()
    if engine1 is not None:
        chatbot1 = get_chatbot(config, engine1)
    else:
        chatbot1 = get_chatbot(config)
    if engine2 is not None:
        chatbot2 = get_chatbot(config, engine2)
    else:
        chatbot2 = get_chatbot(config)
    logger.info('Stage: Prompt Before Start')
    # print('Stage: Prompt Before Start')
    for prompt1 in prompt1_list:
        logger.info(f'Prompt for ChatGPT1: {prompt1}')
        response1 = single_turn_chat(chatbot1, 1, prompt1, output=True)
    for prompt2 in prompt2_list:
        logger.info(f'Prompt for ChatGPT2: {prompt2}')
        response2 = single_turn_chat(chatbot2, 2, prompt2, output=True)
    # print('-' * 30)
    logger.info('Stage: Conversation Start')
    # print('Stage: Conversation Start')
    logger.info(f"ChatGPT2: {response2}")
    # print("ChatGPT2: ", response2, "\n", flush=True)
    for _ in range(max_turn):
        response1 = single_turn_chat(chatbot1, 1, process1(response2))
        response2 = single_turn_chat(chatbot2, 2, process2(response1))
        # if the user wants to end the conversation(contain the ending string), then end the conversation
        if ending(response2):
            break
    return chatbot1, chatbot2

if __name__ == '__main__':
    init()
    multi_chat([], ['随便说句话'], ending=lambda x: '再见' in x, max_turn=5)
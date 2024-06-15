from revChatGPT.V3 import Chatbot
import json
import os
from typing import Callable
from loguru import logger

from usim import init, CONFIG_FILE

def init_config():
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    if 'api_url' in config:
        os.environ['API_URL'] = config['api_url']
    return config

CONFIG = init_config()

def get_chatbot(config: dict = CONFIG, engine: str = None):
    if engine is not None:
        config['engine'] = engine
    chatbot = Chatbot(api_key=config['api_key'], engine=config['engine'])
    return chatbot

def single_turn_chat(chatbot: Chatbot, chat_id: int, query: str, output: bool = True):
    response = chatbot.ask(
        prompt=query,
    )
    if output:
        logger.info(f"ChatGPT{chat_id}: {response}")
        # print(f"ChatGPT{chat_id}: ", response, "\n", flush=True)
    return response

def multi_chat(prompt1_list: str, prompt2_list: str, ending: Callable[[str], bool], process1: Callable[[str], str] = None, process2: Callable[[str], str] = None, max_turn: int = 10):
    if process1 is None:
        process1 = lambda x: x
    if process2 is None:
        process2 = lambda x: x
    config = init_config()
    chatbot1 = get_chatbot(config)
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
    prompt1_list = [
        # f"""
        # 我需要你帮我做一些用户对话模拟，你现在的身份是人工智能对话系统B。你还有一个伙伴用户，Ta的名字是A，请按照以下要求与Ta对话：
        # 1. A有自己喜欢和不喜欢的东西，你不知道Ta具体是什么样的人。
        # 2. 我已经要求A以任意主题开始一段对话。接下来我会把A生成的内容转述给你。获得你的回答后，我会把你的回答转述给A。
        # 3. 我给你的提示会通过“[System]: XXX“与A的回答区分。
        # 4. 请记住，A不一定对你的回答感兴趣，你可以在合适的时机为A推荐、搜索一些商品或是资料，但A也可能因此感到被打扰而感到不舒服，你需要尽量避免这种情况。
        # 5. 你不需要通过括号等形式描述回答被转述或与我进行任何沟通，也不要模仿”[System]: XXX“这样的格式信息，请专注于与A的对话本身即可。
        # 我的要求已经介绍完毕，请你仅仅回复“收到”表示准备好与我进行对话。
        # """
    ]
    prompt2_list = [
        f"""
        我需要你帮我做一些用户对话模拟，你现在的身份是用户A，你需要给自己生成一个自己想要的身份，满足以下要求：
        1. 你需要尽量假装自己是一个真实的人，拥有自己的兴趣偏好，你知道自己喜欢什么不喜欢什么。
        2. 一个真正的人不会对所有东西都感兴趣，所以你一定要有自己不感兴趣的地方。
        3. 一个真正的人拥有丰富的情绪，他不可能永远保持高涨的热情，所以请你为你的身份设置一个当前的情绪状态。
        4. 请你从性别、年龄、性格、职业、爱好、讨厌的事物等至少6个方面生成属于你自己身份的用户画像。
        5. 除此之外，也请你随机生成一个对话的上下文，包括时间、地点、你在进行的事项等至少3方面。
        """,
        f"""
        接下来，你要记住你的名字是A，你将和一个人工智能对话系统B进行聊天，请按照如下要求继续：
        1. 我希望你可以以日常生活，商务活动，或者其他你能想到的任意一个主题开始一段对话，同时请你按照你生成的用户画像和情绪状态来模拟回答。
        2. 我会把你生成的内容转述给B，获得B的回答后，我会把B的回答转述给你，请注意接下来的所有对话我都会完整转述给B。
        3. 我给你的提示会通过“[System]: XXX“与B的回答区分。
        4. 请你记住，如果B的回答并不符合你的需求，也请你表达出来。你不一定要对B的所有回答都很感兴趣。我们需要你扮演的是不同个性的人！
        5. 在B的回答后，你可以调整自己的情绪状态，如对回答感到不满意可以从期待转为平淡或是不满。
        6. 请专注于与B的对话本身，你的回答请不要包含类似“（将B的回答转述给我）”这样的内容，也不要模仿”[System]: XXX“这样的格式信息。
        我的要求已经介绍完毕，现在请你开始吧。
        """
    ]
    process1 = lambda x: '[System]: 以下为用户A生成的内容。\n' + x
    process2 = lambda x: '[System]: 以下为人工智能对话系统B生成的内容。\n' + x + '\n[System]: 如果你想要结束这段对话，请只需回复“谢谢你，再见”。'
    multi_chat(prompt1_list, prompt2_list, ending=lambda x: '谢谢你，再见' in x, process1=process1, process2=process2)
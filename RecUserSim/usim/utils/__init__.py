from usim.chat import CHAT_VERSION

def to_int_list(s: str):
    return [int(x) for x in s.split(',')]

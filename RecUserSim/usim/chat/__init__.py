from usim import CHAT_VERSION

if CHAT_VERSION == '1':
    from .chatV1 import *
elif CHAT_VERSION == '3':
    from .chatV3 import *
elif CHAT_VERSION == '4':
    from .chatV4 import *
else:
    raise ValueError(f'Invalid chat version: {CHAT_VERSION}')
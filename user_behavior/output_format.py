from enum import Enum
from pydantic import BaseModel

class StrategyType(BaseModel):
    pass

class InformationRequest(str, Enum):
    Planning = 'Planning'
    Sequential = 'Sequential'

class OrderV1(str, Enum):
    Depth = 'Depth'
    Breadth = 'Breadth'
    Comprehensive = 'Comprehensive'

class Context(str, Enum):
    High = 'High'
    Low = 'Low'

class Question(str, Enum):
    Broad = 'Broad'
    Specific = 'Specific'

class OrderV2(str, Enum):
    Depth = 'Depth'
    Breadth = 'Breadth'
    DepthBreadth = 'DepthBreadth'
    BreadthDepth = 'BreadthDepth'

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

class StrategyV1(StrategyType):
    information_request: InformationRequest
    order: OrderV1
    context: Context
    question: Question

class StrategyV2(StrategyType):
    order: OrderV2
    feedback: Feedback

class Rating(int, Enum):
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5

class Usefulness(str, Enum):
    High = 'High'
    Moderate = 'Moderate'
    Low = 'Low'

class StrategyV3(StrategyType):
    question_broadness: Rating
    context_dependency: Rating
    feedback: Feedback

class StrategyV4(StrategyType):
    context_dependency: Rating
    explanation: Explanation
    promise: Promise
    feedback: Feedback
    politeness: Politeness
    formality: Formality

class StrategyV5(StrategyType):
    utility: Usefulness
    operability: Usefulness

strategy_list: list[type[StrategyType]] = [StrategyV1, StrategyV2, StrategyV3, StrategyV4, StrategyV5]
template_ids: list[int] = [0, 0, 0, 0, 1]

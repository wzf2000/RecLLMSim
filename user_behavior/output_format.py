from enum import Enum
from pydantic import BaseModel

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

class Feedback(str, Enum):
    NoFeedback = 'NoFeedback'
    Positive = 'Positive'
    Negative = 'Negative'
    Both = 'Both'

class StrategyV1(BaseModel):
    information_request: InformationRequest
    order: OrderV1
    context: Context
    question: Question

class StrategyV2(BaseModel):
    order: OrderV2
    feedback: Feedback

class Rating(int, Enum):
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5

class StrategyV3(BaseModel):
    question_broadness: Rating
    context_dependency: Rating
    feedback: Feedback

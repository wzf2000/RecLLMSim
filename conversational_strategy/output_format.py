from enum import Enum
from pydantic import BaseModel

class Planning(str, Enum):
    PlanOriented = 'PlanningOriented'
    Sequential = 'Sequential'

class OrderV1(str, Enum):
    Depth = 'Depth'
    Breadth = 'Breadth'
    Comprehensive = 'Comprehensive'

class Context(str, Enum):
    High = 'High'
    Low = 'Low'

class Specificity(str, Enum):
    Broad = 'Broad'
    Concrete = 'Concrete'

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
    planning: Planning
    order: OrderV1
    context: Context
    specificity: Specificity

class StrategyV2(BaseModel):
    order: OrderV2
    feedback: Feedback

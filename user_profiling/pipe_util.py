import torch
import random
import numpy as np
from enum import Enum
from typing import Callable, Sequence, Any, Protocol
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from data_util import get_sim_data, get_human_data, ModelType
from log_util import add_log, ExpType

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()

desc_translated = {
    "personality": {
        "内向": "Introverted",
        "外向": "Extroverted",
        "乐观": "Optimistic",
        "悲观": "Pessimistic",
        "冷静": "Calm",
        "友善": "Friendly",
        "坚定": "Determined",
        "幽默": "Humorous",
        "耐心": "Patient",
        "独立": "Independent",
        "细致": "Meticulous",
        "自信": "Confident",
        "上进": "Ambitious",
        "随性": "Spontaneous",
        "其他": "Others"
    },
    "occupation": {
        "金融工作者": "Financial worker",
        "自媒体工作者": "Content creator",
        "医护人员": "Medical staff",
        "学生": "Student",
        "教师": "Educators",
        "警察": "Police officer",
        "律师": "Lawyer",
        "军人": "Military personnel",
        "职业运动员": "Professional athlete",
        "创业者": "Startup founder",
        "艺术工作者": "Artist",
        "厨师": "Chef",
        "工程师": "Engineer",
        "行政人员": "Administrative staff",
        "其他": "Others"
    },
    "daily_interests": {
        "下棋": "Playing chess",
        "养宠物": "Taking care of pets",
        "写作": "Writing",
        "听音乐": "Listening to music",
        "演奏乐器": "Playing instruments",
        "烘焙/烹饪": "Cooking",
        "运动": "Sports",
        "看电视": "Watching TV",
        "手工艺制作": "Crafting",
        "摄影": "Photography",
        "玩游戏": "Playing video games",
        "绘画": "Painting",
        "网购": "Online shopping",
        "阅读": "Reading",
        "其他": "Others"
    },
    "travel_habits": {
        "自驾游": "Self-driving tour",
        "国际旅行": "International travel",
        "经常出差": "Frequently travels for work",
        "看展": "Visiting exhibitions",
        "参观博物馆": "Visiting museums",
        "参观历史遗迹": "Visiting historical sites",
        "自然风光": "Nature-based destinations",
        "喜欢宁静": "Prefers calm and serene locations",
        "家庭旅行": "Family vacations",
        "探亲": "Visiting family",
        "文化旅行": "Cultural trips",
        "美食之旅": "Food tour",
        "豪华住宿": "Luxury accommodation",
        "户外活动": "Outdoor activities",
        "其他": "Others"
    },
    "dining_preferences": {
        "素食": "Vegan",
        "中式美食": "Chinese cuisine",
        "日式美食": "Japanese cuisine",
        "意大利菜": "Italian cuisine",
        "健康饮食": "Healthy food",
        "均衡饮食": "Follows a balanced diet",
        "外卖": "Take-out",
        "尝鲜": "Loves trying new cuisines",
        "快餐": "Fast food",
        "海鲜": "Seafood",
        "烘焙": "Baking",
        "烧烤": "Grilling",
        "街头小吃": "Street food",
        "其他": "Others"
    },
    "spending_habits": {
        "为未来储蓄": "Saves for future",
        "用于旅行": "For travel",
        "购买书籍": "Spends on books",
        "注重消费体验": "Focus on experience",
        "捐款": "Donations",
        "娱乐支出": "Spends on entertainment",
        "房地产投资": "Invests in real estate",
        "时尚支出": "Spends on fashion",
        "科技产品": "Spends on tech gadgets",
        "股票": "Invests in stocks",
        "购买艺术品": "Invests in art pieces",
        "节俭": "Frugal",
        "质量大于数量": "Prefers quality over quantity",
        "高消费": "High spender",
        "其他": "Others"
    },
    "other_aspects": {
        "租房": "Living in a rented house",
        "有房产": "Living in one’s own house",
        "住在宿舍": "Living in a dormitory",
        "养小动物": "Keeping small animals",
        "家庭生活": "Has a family",
        "独自生活": "Lives alone",
        "喜欢参与音乐活动": "Active in music activities",
        "喜欢参与体育活动": "Active in sports activities",
        "喜欢参与志愿活动": "Active in volunteer activities",
        "喜欢参与学术活动": "Active in academic activities",
        "组织社区活动": "Organizes community activities",
        "社交广泛": "Has a large social circle",
        "生活压力大": "Has a high-stress lifestyle",
        "经营自媒体": "Managing personal media platforms",
        "无符合描述": "No description",
        "其他": "Others"
    }
}
human_attributes = {
    "Personality": [
        "Optimistic",
        "Independent",
        "Friendly",
        "Introverted",
        "Calm",
        "Confident",
        "Spontaneous",
        "Meticulous",
        "Extroverted",
        "Ambitious",
        "Patient",
        "Determined",
        "Humorous",
        "Pessimistic"
    ],
    "Daily Interests and Hobbies": [
        "Listening to music",
        "Sports",
        "Playing video games",
        "Reading",
        "Photography",
        "Watching TV",
        "Painting",
        "Online shopping",
        "Cooking",
        "Crafting",
        "Writing",
        "Playing chess",
        "Taking care of pets",
        "Playing instruments"
    ],
    "Travel Habits": [
        "Nature-based destinations",
        "Food tour",
        "Visiting historical sites",
        "International travel",
        "Visiting museums",
        "Prefers calm and serene locations",
        "Cultural trips",
        "Outdoor activities",
        "Visiting exhibitions",
        "Family vacations",
        "Luxury accommodation",
        "Frequently travels for work",
        "Self-driving tour",
        "Visiting family"
    ],
    "Dining Preferences": [
        "Healthy food",
        "Chinese cuisine",
        "Follows a balanced diet",
        "Street food",
        "Grilling",
        "Fast food",
        "Baking",
        "Seafood",
        "Japanese cuisine",
        "Loves trying new cuisines",
        "Vegan",
        "Take-out",
        "Italian cuisine"
    ],
    "Spending Habits": [
        "Focus on experience",
        "Prefers quality over quantity",
        "Spends on entertainment",
        "Spends on tech gadgets",
        "For travel",
        "Saves for future",
        "Spends on fashion",
        "Frugal",
        "Spends on books",
        "Invests in stocks",
        "Donations",
        "Invests in real estate",
        "Invests in art pieces",
        "High spender"
    ]
}

def split_train_test(X: Sequence[Any], y: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(np.array(X), y, test_size=0.2, random_state=42) # type: ignore

def work_sim(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], language: str = 'en', task: str | None = None, **kwargs) -> None:
    X, y = get_sim_data(item, language, task, model_type)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y) # type: ignore
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM, report)

def work_sim2human(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, **kwargs) -> None:
    X_train, y_train = get_sim_data(item, 'zh', task, model_type)
    X_test, y_test = get_human_data(item, task, model_type)
    y = y_train + y_test
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = np.array(X_train)
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM2HUMAN, report)

def work_sim2human2(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, **kwargs) -> None:
    X_train, y_train = get_sim_data(item, 'zh', task, model_type, filtered=True)
    X_test, y_test = get_human_data(item, task, model_type)
    y = y_train + y_test
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM2HUMAN2, report)

def work_sim2human3(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    _, X_test, _, y_test = split_train_test(X_human, y_human)
    X_train, y_train = get_sim_data(item, 'zh', task, model_type, filtered=True)
    print(len(y_train), len(y_test))
    y = y_train + y_test
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = np.array(X_train)
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM2HUMAN3, report)

def work_human(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, samples: int = -1, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X, y = get_human_data(item, task, model_type, data_version, chat_model)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y) # type: ignore
    if samples != -1:
        # sample samples from X_train
        assert 0 < samples <= len(X_train), f"Samples should be between 0 and {len(X_train)}"
        set_seed(42)
        indices = np.random.choice(len(X_train), samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        ckpt_dir_name = f'human_{samples}_{item}'
    else:
        ckpt_dir_name = f'human_{item}'
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, ckpt_dir_name=ckpt_dir_name, **kwargs)
    if samples > 0:
        add_log(item, model_name, ExpType.HUMAN, report, samples=samples)
    else:
        add_log(item, model_name, ExpType.HUMAN, report)

def work_sim4human(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X_sim, y_sim = get_sim_data(item, 'zh', task, model_type)
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    X_train, X_test, y_train, y_test = split_train_test(X_human, y_human)
    X_train = np.concatenate((X_sim, X_train))
    y_train = np.concatenate((y_sim, y_train))
    y = np.concatenate((y_train, y_test))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM4HUMAN, report)

def work_sim4human2(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X_sim, y_sim = get_sim_data(item, 'zh', task, model_type, filtered=True)
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    X_train, X_test, y_train, y_test = split_train_test(X_human, y_human)
    X_train = np.concatenate((X_sim, X_train))
    y_train = np.concatenate((y_sim, y_train))
    y = np.concatenate((y_train, y_test))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM4HUMAN2, report)

def work_sim4human3(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X_sim, y_sim = get_sim_data(item, 'zh', task, model_type, filtered=True)
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    X_train, X_test, y_train, y_test = split_train_test(X_human, y_human)
    # downsample sim data
    _, X_sim, _, y_sim = train_test_split(X_sim, y_sim, test_size=0.2, random_state=42)
    X_train = np.concatenate((X_sim, X_train))
    y_train = np.concatenate((y_sim, y_train))
    y = np.concatenate((y_train, y_test))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.SIM4HUMAN3, report)

def sample(X: np.ndarray, y: np.ndarray, num_sample: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    train_size = len(X)
    sample_size = min(num_sample, train_size)
    indices = np.random.choice(train_size, sample_size, replace=False)
    return X[indices], y[indices]

def work_sim4human4(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, samples: int = -1, ratio: float = 1.0, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    X_sim, y_sim = get_sim_data(item, 'zh', task, model_type, filtered=True)
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    original_size = len(X_sim)
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    X_train, X_test, y_train, y_test = split_train_test(X_human, y_human)
    if samples != -1:
        # sample samples from X_train
        assert 0 < samples <= len(X_train), f"Samples should be between 0 and {len(X_train)}"
        set_seed(42)
        indices = np.random.choice(len(X_train), samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        ckpt_dir_name = f'sim4human4_{samples}_{item}_{ratio}'
    else:
        ckpt_dir_name = f'sim4human4_{item}_{ratio}'
    # downsample sim data
    human_train_size = len(X_train)
    sim_train_size = int(human_train_size * ratio)
    X_sim, y_sim = sample(X_sim, y_sim, sim_train_size)
    X_train = np.concatenate((X_sim, X_train))
    y_train = np.concatenate((y_sim, y_train))
    y = np.concatenate((y_train, y_test))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, ckpt_dir_name=ckpt_dir_name, **kwargs)
    log_name = f'{item}'
    if samples > 0:
        add_log(log_name, model_name, ExpType.SIM4HUMAN4, report, samples=samples, ratio=ratio if sim_train_size < original_size else "full")
    else:
        add_log(log_name, model_name, ExpType.SIM4HUMAN4, report, ratio=ratio if sim_train_size < original_size else "full")

class HotCold(Enum):
    HOT = 'hot'
    COLD = 'cold'
    BOTH = 'both'

def work_sim4human5(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, samples: int = -1, ratio: float = 1.0, hot_cold: HotCold = HotCold.HOT, topk: int = 3, data_version: int = 1, chat_model: str | None = None, **kwargs) -> None:
    if hot_cold == HotCold.HOT:
        attributes = human_attributes[item][:topk]
    elif hot_cold == HotCold.COLD:
        attributes = human_attributes[item][-topk:]
    elif hot_cold == HotCold.BOTH:
        attributes = human_attributes[item][:topk] + human_attributes[item][-topk:]
    else:
        raise ValueError('Invalid hot_cold value')
    item_name = item.split('and')[0].strip().lower().replace(' ', '_')

    def filter(X: Sequence[str], y: list[set[str]]) -> tuple[list[str], list[set[str]]]:
        X_filtered = []
        y_filtered = []
        for i, label in enumerate(y):
            translated_label = [desc_translated[item_name][attr] for attr in label]
            translated_label = set(translated_label)
            if any([attr in translated_label for attr in attributes]):
                X_filtered.append(X[i])
                y_filtered.append(label)
        return X_filtered, y_filtered

    assert model_type != ModelType.HUMAN, "Model type should not be HUMAN for work_sim4human5"
    X_sim, y_sim = get_sim_data(item, 'zh', task, model_type, filtered=True)
    X_sim, y_sim = filter(X_sim, y_sim)
    original_size = len(X_sim)
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    X_human, y_human = get_human_data(item, task, model_type, data_version, chat_model)
    X_train, X_test, y_train, y_test = split_train_test(X_human, y_human)
    if samples != -1:
        # sample samples from X_train
        assert 0 < samples <= len(X_train), f"Samples should be between 0 and {len(X_train)}"
        set_seed(42)
        indices = np.random.choice(len(X_train), samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        ckpt_dir_name = f'sim4human5_{samples}_{item}_{hot_cold.value}_{topk}_{ratio}'
    else:
        ckpt_dir_name = f'sim4human5_{item}_{hot_cold.value}_{topk}_{ratio}'
    # downsample sim data
    human_train_size = len(X_train)
    sim_train_size = int(human_train_size * ratio)
    X_sim, y_sim = sample(X_sim, y_sim, sim_train_size)
    if len(X_sim) == original_size:
        full = True
    else:
        full = False
    log_name = f'{item}'
    X_train = np.concatenate((X_sim, X_train))
    y_train = np.concatenate((y_sim, y_train))
    y = np.concatenate((y_train, y_test))
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    print(f"Number of labels: {len(mlb.classes_)}")
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, ckpt_dir_name=ckpt_dir_name, **kwargs)
    if samples > 0:
        add_log(log_name, model_name, ExpType.SIM4HUMAN5, report, samples=samples, hc=hot_cold.value, topk=topk, ratio=ratio if sim_train_size < original_size else "full")
    else:
        add_log(log_name, model_name, ExpType.SIM4HUMAN5, report, hc=hot_cold.value, topk=topk, ratio="full" if full else ratio)

def work_human2sim(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, **kwargs) -> None:
    X_train, y_train = get_human_data(item, task, model_type)
    X_test, y_test = get_sim_data(item, 'zh', task, model_type)
    y = y_train + y_test
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.HUMAN2SIM, report)

def work_human2sim2(item: str, model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], task: str | None = None, **kwargs) -> None:
    X_train, y_train = get_human_data(item, task, model_type)
    X_test, y_test = get_sim_data(item, 'zh', task, model_type, filtered=True)
    y = y_train + y_test
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    assert isinstance(y, np.ndarray), f"y should be a numpy array, got {type(y)}"
    train_size = len(X_train)
    y_train = y[:train_size]
    y_test = y[train_size:]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    report = work(X_train, y_train, X_test, y_test, item, model_name, mlb.classes_, **kwargs)
    add_log(item, model_name, ExpType.HUMAN2SIM2, report)

def exp(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], work_func: Callable[..., None], only_all: bool = False, items: list[str] | None = None, **kwargs):
    if items is None:
        items = ['Personality', 'Daily Interests and Hobbies', 'Travel Habits', 'Dining Preferences', 'Spending Habits']
    else:
        assert only_all is True, "only_all should be True when items is not None"
        assert all([item in ['Personality', 'Daily Interests and Hobbies', 'Travel Habits', 'Dining Preferences', 'Spending Habits'] for item in items]), "items should be a subset of the default items"
    for item in items:
        work_func(item, model_name, model_type, work, **kwargs)
    if only_all:
        return
    work_func('Travel Habits', model_name, model_type, work, 'travel planning', **kwargs)
    work_func('Dining Preferences', model_name, model_type, work, 'recipe planning', **kwargs)
    work_func('Spending Habits', model_name, model_type, work, 'preparing gifts', **kwargs)
    work_func('Daily Interests and Hobbies', model_name, model_type, work, 'skills learning planning', **kwargs)

def exp_sim(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], language: str | None = None, **kwargs) -> None:
    if language is None:
        languages = ['en', 'zh']
    else:
        languages = [language]
    for lang in languages:
        exp(model_name, model_type, work, work_sim, language=lang, **kwargs)

def exp_sim2human(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim2human, **kwargs)

def exp_sim2human2(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim2human2, **kwargs)

def exp_sim2human3(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim2human3, **kwargs)

def exp_human(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_human, **kwargs)

def exp_sim4human(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim4human, **kwargs)

def exp_sim4human2(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim4human2, **kwargs)

def exp_sim4human3(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim4human3, **kwargs)

def exp_sim4human4(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim4human4, **kwargs)

def exp_sim4human5(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_sim4human5, **kwargs)

def exp_human2sim(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_human2sim, **kwargs)

def exp_human2sim2(model_name: str, model_type: ModelType, work: Callable[..., dict[str, float]], **kwargs) -> None:
    exp(model_name, model_type, work, work_human2sim2, **kwargs)

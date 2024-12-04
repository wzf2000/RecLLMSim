import os
from enum import Enum

log_dir = os.path.join(os.path.dirname(__file__), 'output')

class ExpType(Enum):
    SIM = 'sim'
    SIM2HUMAN = 'sim2human'
    SIM2HUMAN2 = 'sim2human2'
    SIM2HUMAN3 = 'sim2human3'
    HUMAN = 'human'
    SIM4HUMAN = 'sim4human'
    SIM4HUMAN2 = 'sim4human2'
    SIM4HUMAN3 = 'sim4human3'
    SIM4HUMAN4 = 'sim4human4'
    SIM4HUMAN5 = 'sim4human5'
    HUMAN2SIM = 'human2sim'
    HUMAN2SIM2 = 'human2sim2'

def get_log_file(model_name: str, exp_type: ExpType) -> str:
    return os.path.join(log_dir, exp_type.value, f'{model_name}.csv')

def add_log(item: str, model_name: str, exp_type: ExpType, result: dict[str, float], cls: bool = False):
    log_file = get_log_file(model_name, exp_type)
    if not os.path.exists(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            if cls:
                f.write('prediction\tf1_micro\tf1_macro\tf1_weighted\taccuracy\n')
            else:
                f.write('prediction\tf1_micro\tf1_macro\tf1_weighted\taccuracy\thit@1\thit@3\trecall@1\trecall@3\n')
    with open(log_file, 'a') as f:
        if cls:
            f.write(f"{item}\t{result['f1_micro']:.4f}\t{result['f1_macro']:.4f}\t{result['f1_weighted']:.4f}\t{result['accuracy']:.4f}\n")
        else:
            f.write(f"{item}\t{result['f1_micro']:.4f}\t{result['f1_macro']:.4f}\t{result['f1_weighted']:.4f}\t{result['accuracy']:.4f}\t{result['hit_rate_1']:.4f}\t{result['hit_rate_3']:.4f}\t{result['recall_1']:.4f}\t{result['recall_3']:.4f}\n")

import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser, Namespace
from itertools import cycle
from torch.utils.data import DataLoader
from torch.autograd import Function
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from data_util import ModelType, get_sim_data, get_human_data
from evaluate_util import compute_metrics
from pipe_util import set_seed
from predict_lm import MultiLabelDataset


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx: 'GradientReversalFn', x: torch.Tensor, alpha: float):
        # alpha 是控制反转强度的超参数
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: 'GradientReversalFn', grad_output: torch.Tensor):
        # 反向传播时，梯度取反并乘以 alpha
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, alpha)

class BERT_DANN(nn.Module):
    def __init__(self, num_labels: int, model_name: str = 'bert-base-chinese'):
        super(BERT_DANN, self).__init__()

        # 1. 特征提取器 (Backbone)
        self.bert: BertModel = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 通常是 768

        # 2. 用户画像分类器 (主任务: Multi-label)
        # 简单的线性层，也可以加 Dropout 或更多层
        self.label_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

        # 3. 领域判别器 (辅助任务: Binary Classification)
        # 通常是一个 MLP，需要一定的容量来与 BERT 对抗
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # 输出 logits, 0=Sim, 1=Real
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, alpha: float = 1.0):
        # --- Step 1: BERT 提取特征 ---
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 获取 [CLS] token 的 pooled output 作为句向量
        pooled_output = outputs.pooler_output

        # --- Step 2: 主任务预测 (正常流向) ---
        class_logits = self.label_classifier(pooled_output)

        # --- Step 3: 领域判别 (加入梯度反转) ---
        # 这里的 pooled_output 经过 GRL 后，梯度会反转
        reversed_feature = grad_reverse(pooled_output, alpha)
        domain_logits = self.domain_classifier(reversed_feature)

        return class_logits, domain_logits

def evaluation_model(
    model: BERT_DANN,
    loader: DataLoader
) -> torch.Tensor:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            class_logits, _ = model(input_ids, attention_mask, alpha=0.0)  # alpha=0 不需要反转
            probs = torch.sigmoid(class_logits)
            all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)

def train_model(
    model: BERT_DANN,
    loader_sim: DataLoader,
    loader_real: DataLoader,
    test_loader: DataLoader,
    criterion_label: torch.nn.BCEWithLogitsLoss,
    criterion_domain: torch.nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5
):
    y_probs = evaluation_model(model, test_loader)
    metrics = compute_metrics(test_loader.dataset.labels, y_probs.numpy(), more=True)
    log_str = 'Before Training Evaluation metrics:\n'
    log_str += '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(log_str)
    model.train()

    # 假设 loader_sim 和 loader_real 都可以迭代
    # 实际代码中需要处理两个 loader 长度不一致的问题 (通常 sim >> real)
    # 可以使用 itertools.cycle(loader_real) 来循环真实数据

    len_dataloader = len(loader_real)
    loader_sim = cycle(loader_sim)

    for epoch in range(epochs):
        for i, (batch_sim, batch_real) in tqdm(enumerate(zip(loader_sim, loader_real)), total=len_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):

            # --- 0. 动态调整 alpha (可选) ---
            # 训练初期 alpha 小（让特征先成型），后期 alpha 大（加强对抗）
            p = float(i + epoch * len_dataloader) / (epochs * len_dataloader)
            alpha = 2. / (1. + torch.exp(-10 * torch.as_tensor(p))) - 1

            # --- 1. 准备数据 ---
            # Sim 数据: (Input, Label_Profile, Label_Domain=0)
            input_ids_s = batch_sim['input_ids'].to(model.device)
            mask_s = batch_sim['attention_mask'].to(model.device)
            labels_s = batch_sim['labels'].to(model.device)  # Multi-hot vector
            domain_s = torch.zeros(input_ids_s.size(0), 1).to(model.device)  # 0 for Sim

            # Real 数据: (Input, Label_Profile, Label_Domain=1)
            input_ids_r = batch_real['input_ids'].to(model.device)
            mask_r = batch_real['attention_mask'].to(model.device)
            labels_r = batch_real['labels'].to(model.device)
            domain_r = torch.ones(input_ids_r.size(0), 1).to(model.device)  # 1 for Real

            # --- 2. 前向传播 ---
            # 将 Sim 和 Real 拼接在一起进模型 (为了 Batch Normalization 稳定，也可分开跑)
            input_ids = torch.cat([input_ids_s, input_ids_r], dim=0)
            mask = torch.cat([mask_s, mask_r], dim=0)

            class_logits, domain_logits = model(input_ids, mask, alpha=alpha)

            # 拆分结果
            batch_size_s = input_ids_s.size(0)
            class_logits_s = class_logits[:batch_size_s]
            class_logits_r = class_logits[batch_size_s:]

            domain_logits_s = domain_logits[:batch_size_s]
            domain_logits_r = domain_logits[batch_size_s:]

            # --- 3. 计算 Loss ---
            # Task 1: 画像预测 Loss
            # Sim 数据肯定算
            loss_label_s = criterion_label(class_logits_s, labels_s)
            # Real 数据如果你有标签，也可以算（强烈建议算，这是 Supervised DA）
            loss_label_r = criterion_label(class_logits_r, labels_r)
            loss_label = loss_label_s + loss_label_r

            # Task 2: 领域对抗 Loss
            # 必须让 Sim 被预测为 0，Real 被预测为 1
            loss_domain_s = criterion_domain(domain_logits_s, domain_s)
            loss_domain_r = criterion_domain(domain_logits_r, domain_r)
            loss_domain = loss_domain_s + loss_domain_r

            # 总 Loss
            # lambda_weight 可以是一个超参数，或者直接用 alpha 控制
            total_loss = loss_label + loss_domain

            # --- 4. 反向传播 ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 30 == 0:
                with tqdm.external_write_mode(sys.stdout, nolock=False):
                    logger.info(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len_dataloader}], "
                                f"Loss_Label: {loss_label.item():.4f}, Loss_Domain: {loss_domain.item():.4f}, Alpha: {alpha:.4f}")
        # 每个 epoch 结束后评估一次
        y_probs = evaluation_model(model, test_loader)
        metrics = compute_metrics(test_loader.dataset.labels, y_probs.numpy(), more=True)
        log_str = f'Epoch {epoch + 1} Evaluation metrics:\n'
        log_str += '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)

def pipe(item: str, model_name: str, task: str | None = None) -> None:
    set_seed(42)
    X_sim, y_sim = get_sim_data(item, 'zh', task, ModelType.LM, version=5, filtered=True, only="user")
    X_human, y_human = get_human_data(item, task, ModelType.LM, version=2, only="user")
    y = y_human + y_sim
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    human_size = len(X_human)
    y_human = y[:human_size]
    y_sim = y[human_size:]
    X_train, X_test, y_train, y_test = train_test_split(X_human, y_human, test_size=0.2, random_state=42)

    # 找出 y_train 中出现最多的三个 label
    y_sum = y_train.sum(axis=0)
    top3_indices = y_sum.argsort()[-3:][::-1]
    # 计算如果将这三个 label 的概率预测为 0.5, 0.3, 0.2，进行指标计算
    y_probs = np.zeros_like(y_test, dtype=float)
    if len(top3_indices) > 0:
        y_probs[:, top3_indices[0]] = 0.9
    if len(top3_indices) > 1:
        y_probs[:, top3_indices[1]] = 0.8
    if len(top3_indices) > 2:
        y_probs[:, top3_indices[2]] = 0.7
    baseline_metrics = compute_metrics(y_test, y_probs, more=True)
    log_str = f'Baseline metrics for item {item} with model {model_name} (predicting top3 labels with fixed probs):\n'
    log_str += '\n'.join([f"{k}: {v:.4f}" for k, v in baseline_metrics.items()])
    logger.info(log_str)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    real_dataset = MultiLabelDataset(X_train, y_train, tokenizer)
    sim_dataset = MultiLabelDataset(X_sim, y_sim, tokenizer)
    test_dataset = MultiLabelDataset(X_test, y_test, tokenizer)
    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=True)
    sim_loader = DataLoader(sim_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    NUM_LABELS = len(mlb.classes_)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = BERT_DANN(num_labels=NUM_LABELS, model_name=model_name).to(DEVICE)
    model.device = DEVICE  # 方便后续调用

    # 优化器
    # 注意：BERT 层通常需要较小的学习率 (e.g., 2e-5)，分类头可以用大一点 (e.g., 1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Loss 函数
    criterion_label = nn.BCEWithLogitsLoss()  # 多标签分类常用 Loss
    criterion_domain = nn.BCEWithLogitsLoss()  # 二分类 Loss

    # 训练模型
    train_model(
        model,
        sim_loader,
        real_loader,
        test_loader,
        criterion_label,
        criterion_domain,
        optimizer,
        epochs=10
    )

def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train DANN model for user profiling")
    parser.add_argument('-i', '--item', type=str, required=True, help='The profiling item to predict (e.g., Personality)')
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-chinese', help='Pretrained BERT model name')
    parser.add_argument('-t', '--task', type=str, default=None, help='Specific task to filter data (e.g., 旅行规划)')
    return parser.parse_args()

def main():
    args = parse_args()
    pipe(args.item, args.model_name, task=args.task)

if __name__ == "__main__":
    main()

import glob
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, fbeta_score, f1_score
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

from dataset import get_dataset, preprocess_data_lm
from evaluation import compute_metrics_cls, compute_metrics_reg

class_weights = torch.tensor([3.0, 1.0]).to("cuda")  # 调整权重

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class WeightedSampleTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        train_labels = train_dataset.labels
        sample_weights = [5.0 if label == 0 else 1.0 for label in train_labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            len(train_dataset),
            replacement=True
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

def evaluate_lm(model_name: str, train_data: list[dict], test_data: np.ndarray, num_labels: int, regression: bool = False, profile: bool = False) -> dict[str, float]:
    if regression:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression",
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
    X_train, y_train = preprocess_data_lm(train_data, regression, profile=profile)
    X_test, y_test = preprocess_data_lm(test_data, regression, profile=profile)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    train_dataset, val_dataset, test_dataset = get_dataset(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
    output_dir = f'./results/labels_sample_{num_labels}/{model_name}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True
    )
    trainer = WeightedSampleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_cls if not regression else compute_metrics_reg,
    )
    # check if checkpoint exists
    checkpoint = f'{output_dir}/checkpoint-*'
    if len(glob.glob(checkpoint)) > 0:
        print(f"Checkpoint found: {checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()
    results = trainer.evaluate(test_dataset)

    if not regression:
        print(f"Accuracy: {results['eval_accuracy']:.4f}, Weighted F1: {results['eval_f1_weighted']:.4f}")
        if 'eval_recall_0' in results and 'eval_precision_0' in results:
            print(f"Recall 0: {results['eval_recall_0']:.4f}, Precision 0: {results['eval_precision_0']:.4f}")
            print(f"Recall 1: {results['eval_recall_1']:.4f}, Precision 1: {results['eval_precision_1']:.4f}")

            # val_preds = trainer.predict(val_dataset)
            # logits = val_preds.predictions
            # probs_0 = torch.softmax(torch.tensor(logits), dim=-1)[:, 0].numpy()
            # thresholds = np.linspace(0, 1, 100)
            # best_f1 = 0
            # best_threshold = 0.5
            # for thresh in thresholds:
            #     y_pred = (probs_0 > thresh).astype(int)
            #     f1_0 = fbeta_score(val_dataset.labels, y_pred, beta=1, pos_label=0)
            #     if f1_0 > best_f1:
            #         best_f1 = f1_0
            #         best_threshold = thresh

            # test_preds = trainer.predict(test_dataset)
            # logits = test_preds.predictions
            # probs_0 = torch.softmax(torch.tensor(logits), dim=-1)[:, 0].numpy()
            # preds = (probs_0 > best_threshold).astype(int)
            # test_recall_0 = recall_score(test_dataset.labels, preds, pos_label=0)
            # test_recall_1 = recall_score(test_dataset.labels, preds, pos_label=1)
            # test_precision_0 = precision_score(test_dataset.labels, preds, pos_label=0)
            # test_precision_1 = precision_score(test_dataset.labels, preds, pos_label=1)
            # test_acc = accuracy_score(test_dataset.labels, preds)
            # test_f1 = f1_score(test_dataset.labels, preds, average='weighted')
            # print(f"Best threshold: {best_threshold:.4f}, Test recall 0: {test_recall_0:.4f}, Test recall 1: {test_recall_1:.4f}")
            # print(f"Test precision 0: {test_precision_0:.4f}, Test precision 1: {test_precision_1:.4f}")
            # print(f"Test Accuracy: {test_acc:.4f}, Test Weighted F1: {test_f1:.4f}")

    print(f"MSE: {results['eval_mse']:.4f}, RMSE: {results['eval_rmse']:.4f}")
    print(f'GT mean score: {np.mean(y_test):.4f}, Predicted mean score: {results["eval_predict_mean"]:.4f}')
    return results

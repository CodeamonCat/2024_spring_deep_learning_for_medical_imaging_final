import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import torch

from datasets import load_dataset
from evaluate import load
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, TrainingArguments, Trainer
from transformers import Swinv2ForImageClassification, Swinv2Config


def data_collator(examples: list) -> dict:
    pixel_values = torch.stack(
        [example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred) -> dict[str, float]:
    metric = load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(args, train_dataset: Dataset, val_dataset: Dataset,
                id2label: dict[int, str], label2id: dict[str, int]) -> Trainer:
    config = Swinv2Config.from_pretrained(
        args.pretrained_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification")

    model = Swinv2ForImageClassification.from_pretrained(
        args.pretrained_model, config=config, ignore_mismatched_sizes=True)

    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(args.output_dir, 'logs'),
        remove_unused_columns=False,
    )

    tokenizer = AutoFeatureExtractor.from_pretrained(args.pretrained_model)

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    return trainer


def parse_option() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        '2024 spring Deep learning for medical imaging final project',
        add_help=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--data_dir',
                        type=str,
                        default='chest_xray',
                        help='data directory')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='output directory')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default="microsoft/swinv2-tiny-patch4-window8-256",
                        help='pretrained model')
    parser.add_argument('--seed', type=int, default=1216, help='random seed')

    args, unparsed = parser.parse_known_args()
    return args


def save_learning_curve(args, trainer) -> None:
    lr_curve = pd.DataFrame(trainer.state.log_history)
    lr_curve.to_csv(os.path.join(args.output_dir, "image",
                                 "learning_curve.csv"),
                    index=True)


def set_seed(seed: int) -> None:
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def show_report_and_matrix(args, outputs, dataset: Dataset) -> None:
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    labels = dataset.features['label'].names

    # classification report
    print(classification_report(y_true, y_pred, target_names=labels))
    report = classification_report(y_true,
                                   y_pred,
                                   target_names=labels,
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs(os.path.join(args.output_dir, "image"), exist_ok=True)
    report_df.to_csv(os.path.join(args.output_dir, "image",
                                  "classification_report.csv"),
                     index=True)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot().figure_.savefig(
        os.path.join(args.output_dir, "image", "confusion_matrix.png"))


def show_some_result(args, outputs, dataset: Dataset,
                     id2label: dict[str, int]) -> None:
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    index_list = [random.randint(0, len(dataset)) for i in range(25)]
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Some examples of images from the test set", fontsize=20)

    for i, index in enumerate(index_list):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset[index]["image"], cmap=plt.cm.binary)
        plt.xlabel("Label: " + id2label[y_true[index]] + "\nPred: " +
                   id2label[y_pred[index]])
    plt.savefig(os.path.join(args.output_dir, "image", "some_result.png"))


def train_transforms(examples, feature_extractor) -> dict:
    train_transform = transforms.Compose([
        transforms.Resize((feature_extractor.size['height'],
                           feature_extractor.size['width'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean,
                             std=feature_extractor.image_std),
    ])

    examples['pixel_values'] = [
        train_transform(image.convert("RGB")) for image in examples['image']
    ]
    return examples


def val_transforms(examples, feature_extractor) -> dict:
    val_transform = transforms.Compose([
        transforms.Resize((feature_extractor.size['height'],
                           feature_extractor.size['width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean,
                             std=feature_extractor.image_std),
    ])

    examples['pixel_values'] = [
        val_transform(image.convert("RGB")) for image in examples['image']
    ]
    return examples


def main() -> None:
    args = parse_option()
    set_seed(args.seed)

    # load dataset
    train_dataset = load_dataset("imagefolder",
                                 data_dir=os.path.join(args.data_dir, "train"),
                                 split="train")
    val_dataset = load_dataset("imagefolder",
                               data_dir=os.path.join(args.data_dir, "val"),
                               split="train")
    test_dataset = load_dataset("imagefolder",
                                data_dir=os.path.join(args.data_dir, "test"),
                                split="train")

    # transform dataset
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.pretrained_model)
    train_dataset.set_transform(
        partial(train_transforms, feature_extractor=feature_extractor))
    val_dataset.set_transform(
        partial(val_transforms, feature_extractor=feature_extractor))
    test_dataset.set_transform(
        partial(val_transforms, feature_extractor=feature_extractor))

    # set up trainer and training arguments
    id2label = {
        id: label
        for id, label in enumerate(train_dataset.features['label'].names)
    }
    label2id = {label: id for id, label in id2label.items()}
    trainer = get_trainer(args, train_dataset, val_dataset, id2label, label2id)

    # set env
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "offline"

    # train, eval, and test the model
    train_results = trainer.train()
    eval_results = trainer.evaluate()
    outputs = trainer.predict(test_dataset)

    # save best model
    trainer.save_model(os.path.join(args.output_dir, "image"))

    print("==== Train results: ", train_results)
    print("==== Eval results: ", eval_results)
    print("==== Test results: ", outputs.metrics)

    # show classification report, confusion matrix
    show_report_and_matrix(args, outputs, train_dataset)
    show_some_result(args, outputs, test_dataset, id2label)
    save_learning_curve(args, trainer)


if __name__ == "__main__":
    main()

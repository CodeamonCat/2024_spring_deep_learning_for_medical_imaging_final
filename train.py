import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from torchvision.transforms.functional import InterpolationMode


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
                        default=15,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--milestones', type=list, default=[3, 6, 9])
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results_train',
                        help='output directory')
    parser.add_argument('--seed', type=int, default=1216, help='random seed')

    args, unparsed = parser.parse_known_args()
    return args


def plot_learning_curve(logfile_dir, result_lists):
    epoch = range(0, len(result_lists['train_acc']))
    # train_acc_list
    plt.figure(0)
    plt.plot(epoch, result_lists['train_acc'])
    plt.title(f'train_acc_list')
    plt.xlabel('epoch'), plt.ylabel('accuracy')
    plt.savefig(os.path.join(logfile_dir, f'train_acc_list.png'))
    plt.show()
    # train_loss_list
    plt.figure(1)
    plt.plot(epoch, result_lists['train_loss'])
    plt.title(f'train_loss_list')
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.savefig(os.path.join(logfile_dir, f'train_loss_list.png'))
    plt.show()
    # val_acc_list
    plt.figure(2)
    plt.plot(epoch, result_lists['val_acc'])
    plt.title(f'val_acc_list')
    plt.xlabel('epoch'), plt.ylabel('accuracy')
    plt.savefig(os.path.join(logfile_dir, f'val_acc_list.png'))
    plt.show()
    # val_loss_list
    plt.figure(3)
    plt.plot(epoch, result_lists['val_loss'])
    plt.title(f'val_loss_list')
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.savefig(os.path.join(logfile_dir, f'val_loss_list.png'))
    plt.show()


def predict(args, model, test_loader):
    model.eval()
    predictions = list()
    true_labels = list()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            output = model(images)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


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


def show_report_and_matrix(args, y_pred, y_true, dataset: Dataset) -> None:
    labels = dataset.classes

    # classification report
    print(classification_report(y_true, y_pred, target_names=labels))
    report = classification_report(y_true,
                                   y_pred,
                                   target_names=labels,
                                   output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(args.output_dir,
                                  "classification_report.csv"),
                     index=True)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot().figure_.savefig(
        os.path.join(args.output_dir, "confusion_matrix.png"))


def train(args, model, train_loader, val_loader, criterion, optimizer,
          scheduler) -> None:
    best_val_acc = 0.0
    train_loss_list, val_loss_list = list(), list()
    train_acc_list, val_acc_list = list(), list()

    for epoch in range(args.epochs):
        # Train
        train_loss = 0.0
        train_correct = 0.0
        model.train()
        for batch, data in enumerate(train_loader):
            sys.stdout.write(
                f'\r[{epoch + 1}/{args.epochs}] Train batch: {batch + 1} / {len(train_loader)}'
            )
            sys.stdout.flush()
            images, labels = data[0].to(args.device), data[1].to(args.device)
            pred = model(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()

        train_acc = train_correct / len(train_loader.dataset)
        train_loss /= len(train_loader)
        train_acc_list.append(train_acc.cpu())
        train_loss_list.append(train_loss)

        print()
        print(
            f'[{epoch + 1}/{args.epochs}] Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}'
        )

        # Validation
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0

            for data in val_loader:
                images, labels = data[0].to(args.device), data[1].to(
                    args.device)
                output = model(images)
                loss = criterion(output, labels)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred.eq(labels.view_as(pred)).sum().item())

        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(
            f'[{epoch + 1}/{args.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}'
        )

        scheduler.step()

        # Save the best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, 'best_model.pth'))
            print(f'Saved best model with accuracy: {best_val_acc:.5f}')

        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }

        plot_learning_curve(args.output_dir, current_result_lists)


def main():
    args = parse_option()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((272, 272), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                         transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                       transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                        transform=val_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    train(args, model, train_loader, val_loader, criterion, optimizer,
          scheduler)

    # Load the best model for testing
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    predictions, true_labels = predict(args, model, test_loader)

    show_report_and_matrix(args, predictions, true_labels, train_dataset)


if __name__ == "__main__":
    main()

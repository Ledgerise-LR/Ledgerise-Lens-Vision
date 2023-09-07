import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import accuracy_fn
from timeit import Timer as timer
from utils import print_train_time

from typing import Tuple, List, Dict


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y_true=y, y_pred=y_pred)

        train_loss += loss
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            y_pred = torch.round(torch.sigmoid(y_logits))

            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y_true=y, y_pred=y_pred)

            test_loss += loss
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int,
    optimizer: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device,
):
    start_time = timer()

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}\n-------------------------")

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    end_time = timer()

    print_train_time(start_time, end_time, device)

    return results

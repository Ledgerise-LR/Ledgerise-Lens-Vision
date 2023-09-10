import torch
import torchvision
import matplotlib.pyplot as plt
from typing import List
import numpy as np


def pred_and_plot_on_custom_data(
    model: torch.nn.Module,
    image,  # type: ignore
    transform: torchvision.transforms,  # type: ignore
    class_names: List[str] = None,  # type: ignore
    device: str = None,  # type: ignore
):
    # custom_image = torchvision.io.read_image(str(image_path)) / 255

    custom_image_transformed = transform(image)  # type: ignore

    pred_logits = model(custom_image_transformed.unsqueeze(dim=0))

    pred_probs = torch.softmax(pred_logits, dim=1)

    pred_label = pred_probs.argmax(dim=1)

    # plt.imshow(custom_image_transformed.permute(1, 2, 0))
    # plt.axis("off")

    title = f"Pred: {class_names[pred_label]} | Prob: {(pred_probs[pred_label] * 100):.1f} %"

    # plt.title(title)

    return class_names[pred_label]

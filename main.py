from process_custom import process_custom
from preprocess.preprocess import preprocess
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from prediction import pred_and_plot_on_custom_data
import model_builder
import torch
from torchvision import transforms
import cv2

HIDDEN_UNITS = 16
class_names = ["not_parcel", "parcel"]

model = model_builder.LedgeriseLens(
    input_channels=3, hidden_units=HIDDEN_UNITS, output_channels=len(class_names)
)

model.load_state_dict(torch.load(f="models/LedgeriseLensV3.pth"))

transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ]
)


def main():
    preprocess_out = preprocess()

    img_rgb_shape = preprocess_out["img_rgb"].shape

    canvas_black = np.zeros(img_rgb_shape, dtype=np.uint8)

    print(f"Shape of img rgb: {img_rgb_shape}")

    for i in preprocess_out["rect_images"]:
        rect_shape = i["extracted_rectangle"].shape
        x, y, w, h = i["x"], i["y"], i["w"], i["h"]
        print(f"Shape of the extracted rectangle: {rect_shape}")
        print(f"X: {x}, Y: {y}, W: {w}, H: {h}")  # type: ignore

        rect_image = i["extracted_rectangle"]

        pred_label = pred_and_plot_on_custom_data(
            model=model,
            image=cv2.cvtColor(rect_image, cv2.COLOR_rgb),
            transform=transform,  # type: ignore
            class_names=class_names,
        )

        print(pred_label)
        # canvas_black[y : y + h, x : x + w] = rect_image

    plt.imshow(canvas_black)
    plt.axis("off")
    plt.title("Only objects image not classified yet.")
    plt.show()


if __name__ == "__main__":
    main()

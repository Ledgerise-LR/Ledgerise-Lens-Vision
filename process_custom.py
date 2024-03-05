# from preprocess.preprocess import preprocess
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from prediction import pred_and_plot_on_custom_data
import model_builder
import torch
from torchvision import transforms
from preprocess.preprocess_improved import preprocessv2

# from PIL import Image
# from io import StringIO

HIDDEN_UNITS = 16
class_names = ["not_parcel", "parcel"]

model = model_builder.LedgeriseLens(
    input_channels=3, hidden_units=HIDDEN_UNITS, output_channels=len(class_names)
)

model.load_state_dict(torch.load(f="./models/LedgeriseLensV4.pth"))

transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
    ]
)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH)


def process_custom(img, bounds):
    user_info = ""
    found_status = "false"

    # while True:
    # img = np.asarray(bytearray(img_base64), dtype=np.uint8)
    # img = cv2.imdecode(img, 0)

    # ret, img = cap.read()
    # img = cv2.imread("./preprocess/data/img2.png")
    preprocess_out = preprocessv2(img=img)

    img_rgb = preprocess_out[1]
    # img_rgb_shape = preprocess_out[1].shape

    # canvas_black = np.zeros(img_rgb_shape, dtype=np.uint8)

    coordinates = []
    qr_coordinates = []

    max_pred_prob = 0

    for i in preprocess_out[0]:
        x, y, w, h = i[0], i[1], i[2], i[3]

        rect_image = img_rgb[
            y : y + h,
            x : x + w,
        ]

        rect_image_resized = np.array(rect_image, dtype=np.uint8)

        pred_label, pred_prob = pred_and_plot_on_custom_data(
            model=model,
            image=rect_image_resized,
            transform=transform,  # type: ignore
            class_names=class_names,
        )

        if pred_label == "parcel" and x < bounds["x"] and bounds["x"] < x + w and h < bounds["y"] and bounds["y"] < y + h:
            if (
                rect_image.shape[0] == img_rgb.shape[0]
                and rect_image.shape[1] == img_rgb.shape[1]
            ):
                pass
            else:
                if (pred_prob > max_pred_prob):
                    max_pred_prob = pred_prob
                    found_status = "true"
                    coordinates = [x, w, y, h]
                    # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 5)
                    # cv2.putText(
                    #     img_rgb,
                    #     "parcel",
                    #     (x + MARGIN, y + MARGIN),
                    #     cv2.FONT_HERSHEY_COMPLEX,
                    #     1,
                    #     (255, 0, 0),
                    #     2,
                    #     cv2.LINE_AA,
                    # )
                else:
                    pass
        else:
            pass

    max_pred_prob = 0
    return found_status, coordinates
    # cv2.imshow("frame", cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    # if cv2.waitKey(1) == ord("q"):
    #     break

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision import transforms

from PIL import Image

from models.bisenet import BiSeNet


cityscapes_colormap = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32),    # bicycle
    19: (0, 0, 0),        # unlabeled (often ignored)
}

def load_image_as_tensor(image_path):
    try:
        image = Image.open(image_path).convert('RGB')

    except FileNotFoundError:
        print(f"Error: the image file '{image_path}' was not found")
        exit()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.CenterCrop(size=(512, 1024))
    ])

    image_tensor = to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def output_to_image(output, colormap):
    if torch.is_tensor(output):
        output = output.squeeze().cpu().numpy()

    image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)

    for label_id, color in colormap.items():
        image[output == label_id] = color

    return Image.fromarray(image)


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bisenet = BiSeNet('STDCNet813', 19)
    weights_path = './checkpoints/STDC1-Seg/model_maxmIOU50.pth'
    bisenet.load_state_dict(torch.load(weights_path))

    bisenet.eval()

    image_tensor = load_image_as_tensor('./berlin_000000_000019_leftImg8bit.png')
    print('Image tensor shape:', image_tensor.shape)

    bisenet = bisenet.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = bisenet(image_tensor)

    size = image_tensor.size()[-2:]

    logits = F.interpolate(
        output[0],
        size=size,
        mode='bilinear',
        align_corners=True
    )

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    label = output_to_image(preds, cityscapes_colormap)
    label.save('result.png')

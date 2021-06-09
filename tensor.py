import math

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


def main():
    sample_image = cv2.imread("sample.jpg")  # BGR read
    image_pil = Image.fromarray(sample_image)
    cv2.imshow("sample_original", sample_image)

    # image augmentation. hue shift?
    transform_pos = get_transforms_pos()
    transform_neg = get_transforms_neg()

    image_aug_crop = None
    x1, y1, x2, y2 = 134, 52, 408, 778
    cx, cy, w, h = (x2 + x1)//2, (y2 + y1)//2, x2 - x1, y2 - y1
    original_target = sample_image[y1:y2, x1:x2, :]

    for i in range(100):
        random_number = np.random.randn()
        if random_number > 0:
            image_aug_pil = transform_pos(image_pil)
        else:
            image_aug_pil = transform_neg(image_pil)

        image_aug = np.array(image_aug_pil)
        #image_aug[y1-30:y2-30, x1-30:x2-30, :] = original_target
        cv2.imshow("cropped_augmented_image", image_aug)
        cv2.waitKey(16)

    search_factor = 3
    sz_crop = math.ceil(math.sqrt(w * h)) * search_factor



def get_transforms_pos():
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5,
                               contrast=0.5,
                               saturation=0.5,
                               hue=(0.2, 0.5))
    ])
    return transform


def get_transforms_neg():
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=(-0.5, -0.2))
    ])
    return transform


if __name__ == '__main__':
    main()

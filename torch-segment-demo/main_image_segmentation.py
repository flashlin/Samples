# https://github.com/facebookresearch/segment-anything
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2

from image_segmentation_utils import save_image_segmentation, get_image_segmentation_model, ImageSegmentation
from io_utils import query_sub_files


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def save_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    idx = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        save_path = f'./output/ann_{idx}.jpg'

        #m_uint8 = (m * 255).astype(np.uint8)
        #cv2.imwrite(save_path, m_uint8)
        masked_img = img.copy()
        masked_img[~m] = [1, 1, 1, 0]  # 將非 `m` 的部分設為完全透明
        cv2.imwrite(save_path, (masked_img * 255).astype(np.uint8))
        idx += 1

        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_imgae(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()


# def get_mask(image):
#     predictor = SamPredictor(sam)
#     predictor.set_image(image)
#
#     input_point = np.array([[500, 375]])
#     input_label = np.array([1])
#
#     masks, scores, logits = predictor.predict(
#         # point_coords=input_point,
#         # point_labels=input_label,
#         multimask_output=True,
#     )
#
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca())
#         show_points(input_point, input_label, plt.gca())
#         plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.show()


# def generate_mask(image):
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     masks = mask_generator.generate(image)

#torch.cuda.set_per_process_memory_fraction(0.9)
idx = 0
image = ImageSegmentation()
image.save_segmentation('./data/train_segmentation/CasSmallPic/CAS_promo_banner05_en.jpg', 'output/test', 0)

# for image_file in query_sub_files('./data/train_segmentation', ['.jpg', '.png', '.gif']):
#     print(f'{image_file}')
#     idx = save_image_segmentation(sam, image_file, './output/segmentation', idx)

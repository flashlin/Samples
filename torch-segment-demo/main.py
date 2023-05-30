# https://github.com/facebookresearch/segment-anything
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


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



def save_anns2(image, anns):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = sorted(anns, key=(lambda item: (item['bbox'][1], item['bbox'][0])), reverse=False)
    idx = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        x, y, w, h = ann['bbox']
        #print(f'{ann=}')
        save_path = f'./output/ann_{idx}.jpg'
        masked_img = image.copy()
        masked_img[~m] = [1, 1, 0]  # 將非 `m` 的部分設為完全透明
        cropped_img = masked_img[y:y + h, x:x + w]
        cv2.imwrite(save_path, (cropped_img * 255).astype(np.uint8))
        idx += 1


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_imgae(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()


model_type = 'vit_l'  # vit_h / vit_l / vit_b
sam_checkpoint = "sam_vit_l_0b3195.pth"

device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=f"./models/{sam_checkpoint}")
sam.to(device=device)


def get_mask(image):
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        # point_coords=input_point,
        # point_labels=input_label,
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def get_mask2(image):
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    masks2 = mask_generator_2.generate(image)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    # show_anns(masks2)
    # save_anns(masks2)
    save_anns2(image, masks2)
    plt.axis('off')
    #plt.show()


def generate_mask(image):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)


image = read_image('./images/sbotop-deposit-bonus.jpg')
# show_imgae(image)
get_mask2(image)

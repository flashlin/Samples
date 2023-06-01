# https://github.com/facebookresearch/segment-anything
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

model_type = 'vit_l'  # vit_h / vit_l / vit_b
sam_checkpoint = "sam_vit_l_0b3195.pth"
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=f"./models/{sam_checkpoint}")
sam.to(device=device)


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_annotations(image, annotations, output_dir: str, idx: int = 0):
    if len(annotations) == 0:
        return idx
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_annotations = sorted(annotations, key=(lambda item: (item['bbox'][1], item['bbox'][0])), reverse=False)
    for ann in sorted_annotations:
        m = ann['segmentation']
        x, y, w, h = ann['bbox']
        save_path = f'{output_dir}/ann_{idx}.jpg'
        masked_img = image.copy()
        masked_img[~m] = [1, 1, 0]  # 將非 `m` 的部分設為完全透明
        cropped_img = masked_img[y:y + h, x:x + w]
        cv2.imwrite(save_path, (cropped_img * 255).astype(np.uint8))
        idx += 1
    return idx


def save_image_segmentation(image_path: str, output_dir: str, idx: int = 0):
    image = read_image(image_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(image)
    return save_annotations(image, masks, output_dir, idx)

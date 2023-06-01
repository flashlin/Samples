import cv2
import numpy as np
from segment_anything import SamPredictor, SamAutomaticMaskGenerator

from main import read_image, sam


def save_annotations(image, annotations, output_dir: str):
    if len(annotations) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_annotations = sorted(annotations, key=(lambda item: (item['bbox'][1], item['bbox'][0])), reverse=False)
    idx = 0
    for ann in sorted_annotations:
        m = ann['segmentation']
        x, y, w, h = ann['bbox']
        save_path = f'{output_dir}/ann_{idx}.jpg'
        masked_img = image.copy()
        masked_img[~m] = [1, 1, 0]  # 將非 `m` 的部分設為完全透明
        cropped_img = masked_img[y:y + h, x:x + w]
        cv2.imwrite(save_path, (cropped_img * 255).astype(np.uint8))
        idx += 1


def save_image_segmentation(image_path: str, output_dir: str):
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
    save_annotations(image, masks, output_dir)

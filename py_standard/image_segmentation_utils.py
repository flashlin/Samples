# https://github.com/facebookresearch/segment-anything
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_annotations(image, annotations, output_dir: str, idx: int = 0):
    if len(annotations) == 0:
        return idx
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_annotations = sorted(annotations, key=(lambda item: (item['bbox'][1], item['bbox'][0])), reverse=False)
    for i, ann in enumerate(sorted_annotations):
        m = ann['segmentation']
        x, y, w, h = ann['bbox']
        save_path = f'{output_dir}/ann_{idx}.jpg'
        masked_img = image.copy()
        masked_img[~m] = [1, 1, 0]  # 將非 `m` 的部分設為完全透明
        # print(f'{idx=} {x=} {y=} {w=} {h=}')
        y = int(y)
        x = int(x)
        w = int(w)
        h = int(h)
        cropped_img = masked_img[y:y + h, x:x + w]
        if w <= 1 or h <= 1:
            continue
        if w * h <= 20 * 20:
            continue
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cropped_img.astype(np.uint8))
        idx += 1
    return idx


def get_image_segmentation_model(device='cuda'):
    model_type = 'vit_l'  # vit_h / vit_l / vit_b
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    sam = sam_model_registry[model_type](checkpoint=f"./models/{sam_checkpoint}")
    sam.to(device=device)
    return sam


def save_image_segmentation(sam, image_path: str, output_dir: str, idx: int = 0):
    image = read_image(image_path)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # 每個邊的分割點數量
        pred_iou_thresh=0.86,  # 0.86 生成遮罩時所使用的預測IOU閾值, 設為0.8或更低，以使更多的預測被考慮生成遮罩
        stability_score_thresh=0.92,  # 控制了生成遮罩時所使用的穩定性分數閾值
        crop_n_layers=1,  # 裁剪操作的層數
        crop_n_points_downscale_factor=2,  # 裁剪操作中下採樣點的數量因子, 增加此值可以使裁剪更加精確，但同時也會增加計算成本
        min_mask_region_area=256,  # 遮罩區域的最小面積
    )

    masks = mask_generator.generate(image)
    return save_annotations(image, masks, output_dir, idx)

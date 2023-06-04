# https://github.com/facebookresearch/segment-anything
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

from io_utils import get_filename, split_filename


def is_same_image(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    if image1.mode != image2.mode or image1.size != image2.size:
        return False
    pixel_pairs = zip(image1.getdata(), image2.getdata())
    if any(p1 != p2 for p1, p2 in pixel_pairs):
        return False
    return True


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if w >= 1:
        w = 0.99
    if h >= 1:
        h = 0.99
    return (x, y, w, h)


def convert_labelimg_annotation_xml_to_txt(xml_file_path, classes, output_dir):
    # classes = ['person']
    # directory = os.path.dirname(xml_file_path)
    xml_filename = os.path.basename(xml_file_path)
    txt_filename = xml_filename[:-4] + '.txt'
    txt_file_path = output_dir + '/' + txt_filename
    if os.path.exists(txt_file_path):
        return
    with open(xml_file_path, "r", encoding='UTF-8') as in_file:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(txt_file_path, "w", encoding='UTF-8') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            out_file.truncate()
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_annotations(image, annotations, full_filename: str, output_dir: str):
    if len(annotations) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    filename, file_ext = split_filename(full_filename)

    sorted_annotations = sorted(annotations, key=(lambda item: (item['bbox'][1], item['bbox'][0])), reverse=False)
    idx = 0
    for i, ann in enumerate(sorted_annotations):
        m = ann['segmentation']
        x, y, w, h = ann['bbox']
        save_path = os.path.join(output_dir, f'/{filename}_ann_{idx}.{file_ext}')
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
    return


class ImageSegmentation:
    def __init__(self, device='cuda'):
        model_type = 'vit_l'  # vit_h / vit_l / vit_b
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        sam = sam_model_registry[model_type](checkpoint=f"./models/{sam_checkpoint}")
        sam.to(device=device)
        predictor = SamPredictor(sam)
        self.sam = sam
        self.predictor = predictor

    def save_segmentation(self, image_path: str, output_dir: str, idx: int = 0):
        full_filename = os.path.basename(image_path)

        image = read_image(image_path)
        self.predictor.set_image(image)

        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,  # 每個邊的分割點數量
            pred_iou_thresh=0.86,  # 0.86 生成遮罩時所使用的預測IOU閾值, 用於判斷預測的區域是否與真實標籤重疊得足夠多
            stability_score_thresh=0.7,  # 0.92 穩定性分數閾值, 例如概率分數、置信度分數或方差等來進行判斷
            crop_n_layers=1,  # 裁剪操作的層數
            crop_n_points_downscale_factor=2,  # 裁剪操作中下採樣點的數量因子, 增加此值可以使裁剪更加精確，但同時也會增加計算成本
            min_mask_region_area=256,  # 如果區域的面積小於 min_mask_region_area，則該區域會被視為無效區域，被過濾掉或視為背景
        )

        masks = mask_generator.generate(image)
        return save_annotations(image, masks, full_filename, output_dir)

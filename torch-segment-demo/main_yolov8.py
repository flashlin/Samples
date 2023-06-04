# ultralytics
# pip install PyQt5
# pip install pyqt5-tools
# pip install lxml
# pip install labelimg
# https://app.roboflow.com/flash-elbmq/banner-akbct/upload
from image_segmentation_utils import convert_labelimg_annotation_xml_to_txt

# git clone https://github.com/tzutalin/labelImg
# pyrcc5 -o resources.py resources.qrc
# 將 resources.py 複製到 lib folder


# git clone https://github.com/ultralytics/ultralytics.git
# pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 到這個下載預訓練模型
# https://github.com/ultralytics/ultralytics
# 下載好預訓練模型放置到 ultralytics/ultralytics 目錄下
# 將數據放置到ultralytics/ultralytics/dataset下
# 修改yaml文件，在ultralytics/ultralytics/datasets底下新建act.yaml
# 直接用預訓練模型進行預測，在ultralytics/ultralytics/runs/detect/下直接生成結果 source可以是圖片、視頻、文件夾
# yolo predict model=yolov8x.pt source='dataset/only-tarin-wheel-datasets'
# yolo train data=datasets/act.yaml model=yolov8x.pt epochs=3 lr0=0.01

convert_labelimg_annotation_xml_to_txt('./data/labelimg/CAS_promo_banner05_en.xml',
                                       classes=['A', 'B', 'C', 'D', 'E', 'F',
                                                'G', 'H', 'I', 'J', 'K', 'L',
                                                'M', 'N', 'O', 'P', 'Q', 'R',
                                                'S', 'T', 'U', 'V', 'W', 'X',
                                                'Y', 'Z', '0', '1', '2', '3',
                                                '4', '5', '6', '7', '8', '9'],
                                       output_dir='./data/yolo/train/labels')


# yolo export model=best.pt format=onnx  # export custom trained model

# yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" conf=0.3, iou=0.5 show tracker="bytetrack.yaml"
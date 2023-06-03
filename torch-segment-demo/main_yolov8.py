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

# https://github.com/tzutalin/labelImg

convert_labelimg_annotation_xml_to_txt('./data/labelimg/CAS_promo_banner05_en.xml',
                                       classes=['A', 'B', 'C', 'D', 'E', 'F',
                                                'G', 'H', 'I', 'J', 'K', 'L',
                                                'M', 'N', 'O', 'P', 'Q', 'R',
                                                'S', 'T', 'U', 'V', 'W', 'X',
                                                'Y', 'Z', '0', '1', '2', '3',
                                                '4', '5', '6', '7', '8', '9'],
                                       output_dir='./data/yolo/train/labels')

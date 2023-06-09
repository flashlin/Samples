
```
pip install labelme
git clone https://github.com/wkentaro/labelme
pip install imgviz
pip install -r .\requirements-dev.txt
```

# 轉換為 coco 格式
$ python labelme2coco.py <data> <data_output> --labels <label.txt path>
# 轉換為 VOC 格式
$ python labelme2voc.py <data> <data_output> --labels <label.txt path>
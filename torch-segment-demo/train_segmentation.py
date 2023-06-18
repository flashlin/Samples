from image_annotations_utils import ImageAnnotationsDataset
from image_resnet50 import ImageMasks
from image_utils import load_image

image_dataset = ImageAnnotationsDataset("data/yolo/train")

image_masker = ImageMasks(image_dataset.classes)
image_masker.train(image_dataset, batch_size=8, num_epochs=100)

input_image = load_image('data/yolo/train/images/ace45-my-zh-cn.jpg')
shot_images, segmented_image = image_masker.infer(input_image)

#for (shot_image, mask_image), label in shot_images:
    #shot_image.show()
    #print(f'{label=}')

if segmented_image is not None:
    segmented_image.show()
else:
    print("Not Found")

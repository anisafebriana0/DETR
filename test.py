import torch
import supervision as sv
import transformers
import pytorch_lightning
import os
import torchvision

dataset = "C:/Dataset/paralysis face.v7i.coco"

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")

#class for load dataset
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__( self, image_directory_path:str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self , idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations':annotations}
        encoding = self.image_processor(images = images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    

from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

print("Number of train image : ", len(TRAIN_DATASET))
print("Number of validation image : ", len(VAL_DATASET))
print("Number of test image : ", len(TEST_DATASET))

# prepare data before training

def collate_fn(batch):
    pixel_vales = []

    for item in batch:
        pixel_vales.append(item[0])

    
    encoding = image_processor.pad(pixel_vales, return_tensors="pt")

    labels = []
    for item in batch:
        labels.append(item[1])

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

categories = TRAIN_DATASET.coco.cats
print("Categories:")
print(categories)


id2label = {}
for k, v, in categories.items():
    id2label[k] = v['name']

print("id2label", id2label)
print(len(id2label))

print("===========")
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn,batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn,batch_size=4)



import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

MODEL_PATH = "./DETR-My-Model-1"
device = "cpu"
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


print(model)

#============================================================
box_annotator = sv.BoxAnnotator()

import cv2
#load image test folder
image_path ="C:/Dataset/paralysis face.v7i.coco/test/10361_bmp.rf.aeb0a2877123415ed165d8e052bc3997.jpg"
image = cv2.imread(image_path)

CONFIDENCE_THRESHOLD = 0.2

with torch.no_grad():

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold = CONFIDENCE_THRESHOLD,
        target_sizes= target_sizes)[0]
#create the output image
detections = sv.Detections.from_transformers(transformers_results=results)
labels = [f"{id2label[class_id]} {confidence:.2f}" for _,confidence , class_id, _ in detections]
image_with_detection = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

cv2.imshow("image with detection", image_with_detection)
cv2.imshow("img", image)
cv2.waitKey(0)


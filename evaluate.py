import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
import numpy as np

dataset = "C:/Dataset/paralysis face.v7i.coco"
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")

# Load the trained model and processor
MODEL_PATH = "./DETR-My-Model-1"
device = "cpu"

# Load model and processor
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(device)
image_processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

# Load test dataset (update TEST_DIRECTORY if needed)
TEST_DIRECTORY = "C:/Dataset/paralysis face.v7i.coco/test"
ANNOTATION_FILE_NAME = "_annotations.coco.json"

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
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)



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
TEST_DATALOADER = DataLoader(TEST_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=False)
# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch['pixel_mask'].to(device)
            labels = batch['labels']

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            logits = outputs.logits

            # Decode predictions
            for i, logit in enumerate(logits):
                preds = torch.argmax(logit, dim=-1)
                targets = labels[i]["labels"]

                # Calculate correct predictions
                correct += (preds == targets).sum().item()
                total += len(targets)

    accuracy = correct / total * 100
    return accuracy

# Perform evaluation
accuracy = evaluate_model(model, TEST_DATALOADER, device)

# Save results
save_dir = "./evaluation_results"
os.makedirs(save_dir, exist_ok=True)
results_path = os.path.join(save_dir, "results.json")

results = {
    "accuracy": accuracy
}

with open(results_path, "w") as f:
    json.dump(results, f)

print(f"Accuracy on the test set: {accuracy:.2f}%")
print(f"Results saved to {results_path}")

# Plotting (optional)
def plot_results(accuracy):
    plt.figure(figsize=(6, 4))
    plt.bar(["Test Set"], [accuracy], color='blue')
    plt.ylabel("Accuracy (%)")
    plt.title("Evaluation Results")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

plot_results(accuracy)

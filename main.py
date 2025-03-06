import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Define Pascal VOC classes (20 classes from VOC dataset)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Paths to dataset directories
VOC_ANNOTATIONS_PATH = "VOCdevkit/VOC2012/Annotations/"  # XML files
YOLO_LABELS_PATH = "VOCdevkit/VOC2012/labels/"  # Output folder
VOC_IMAGE_PATH = "VOCdevkit/VOC2012/JPEGImages/"

# Create output directory if it doesn't exist
os.makedirs(YOLO_LABELS_PATH, exist_ok=True)


def convert_voc_to_yolo(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    with open(output_txt, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in VOC_CLASSES:
                continue
            class_id = VOC_CLASSES.index(class_name)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center = (xmin + xmax) / (2.0 * img_width)
            y_center = (ymin + ymax) / (2.0 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


xml_files = [f for f in os.listdir(VOC_ANNOTATIONS_PATH) if f.endswith(".xml")]

for xml_file in tqdm(xml_files, desc="Converting VOC 2012 → YOLO Format"):
    xml_path = os.path.join(VOC_ANNOTATIONS_PATH, xml_file)
    txt_output = os.path.join(YOLO_LABELS_PATH, xml_file.replace(".xml", ".txt"))
    convert_voc_to_yolo(xml_path, txt_output)

print("Pascal VOC 2012 annotations converted to YOLO format successfully!")


# Define the YOLO v1 Architecture
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 28 * 28, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, 1470)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fully_connected(x)
        return x.view(-1, 7, 7, 30)


transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace(".jpg", ".txt"))

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.zeros((7, 7, 30))  # Placeholder for labels


dataset = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16, shuffle=True)
test_dataloader = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16,
                             shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv1().to("cpu")

criterion = nn.MSELoss()  # Placeholder loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

num_epochs = 50

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    model.train()
    epoch_loss = 0
    for images, labels in dataset:
        images, labels = images.to("cpu"), labels.to("cpu")
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

model.eval()
test_loss = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to("cpu"), labels.to("cpu")
        predictions = model(images)
        loss = criterion(predictions, labels)
        test_loss += loss.item()
print(f"Test Loss: {test_loss:.4f}")

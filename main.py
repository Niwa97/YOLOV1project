import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import models

#Transforming VOC2012 classes to YOLO format - data set with 20 classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

#Paths set for google collab enfironment
VOC_ANNOTATIONS_PATH = "/content/sample_data/VOCdevkit/VOC2012/Annotations"  
YOLO_LABELS_PATH = "/content/sample_data/VOCdevkit/VOC2012/labels" 
VOC_IMAGE_PATH = "/content/sample_data/VOCdevkit/VOC2012/JPEGImages"


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

#Implementation of YOLO v1 model  
#The model is implemented as a subclass of torch.nn.Module
#Implemented according to "You Only Look Once:Unified, Real-Time Object Detection", Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
#input size 448x448x30
#output size 7x7x30
#2 main layer parts convolutional and fully connected
class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
#Convolutional layer to extract image data
#LeakyReLU recommended for sparse training (that is f(x) = x, x >= 0 and 0.1x x <0)
#filter sizes varying from 1 to 3
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
#Fully connected layer for feature extraction
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

#Preprocessing part
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

#Transforms images data to YOLOv1 output
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_files)
    #Loads the label data from YOLO label files, assigns the box information and places it into the grid cell
    #Object and class presence considered as data in labels
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace(".jpg", ".txt"))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = torch.zeros((7, 7, 30))  
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                grid_x = int(x_center * 7)
                grid_y = int(y_center * 7)
                labels[grid_y, grid_x, 0] = x_center
                labels[grid_y, grid_x, 1] = y_center
                labels[grid_y, grid_x, 2] = width
                labels[grid_y, grid_x, 3] = height
                labels[grid_y, grid_x, 4] = 1  
                labels[grid_y, grid_x, 5 + class_id] = 1  
        return image, labels

#Loss function prepared for YOLO models
#In v1/v2 version loss function is fairly simple new version uses much more sophisticated approach
#We sum three loss terms (5 in paper): 
#Localization loss - Control errors in box predictions, penalizes bad localization of center of cells and inaccurate height and width.
#Confidence loss - Controllss the confidence of the most accurate box and tries to make confidence score close to 0
#Classification loss - Penalize incorrect class predictions

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1, self.S, self.S, (self.C + self.B * 5))
        coord_mask = targets[..., 4] > 0  
        noobj_mask = targets[..., 4] == 0   
        localization_loss = self.lambda_coord * torch.sum(coord_mask.unsqueeze(-1) * (torch.square(predictions[..., :2] - targets[..., :2]) + torch.square(torch.sqrt(predictions[..., 2:4].clamp(min=1e-6)) - torch.sqrt(targets[..., 2:4].clamp(min=1e-6)))))
        confidence_loss = torch.sum(self.lambda_noobj * noobj_mask.unsqueeze(-1) * torch.square(predictions[..., 4:5] - targets[..., 4:5]))
        confidence_loss += torch.sum(coord_mask.unsqueeze(-1) * torch.square(predictions[..., 4:5] - targets[..., 4:5]))
        classification_loss = torch.sum(coord_mask.unsqueeze(-1) * torch.square(predictions[..., 5:] - targets[..., 5:]))
        loss = localization_loss + confidence_loss + classification_loss
        return loss

dataset = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16, shuffle=True)
test_dataloader = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16, shuffle=True)

#GPU calculations are necessary - CPUs caclulations are extremally slow
model = YOLOv1().to("cuda")

#For VOC2012 simple L^2 norm is working a faster than YoloLoss and for testing reasons work fine but for application reasons standard error criterion will definitely not suffice
# ( on test data loss function can be around 3 - needed 0.03)
#Mean square error seems to be a better than L^1
#Yolo loss should provide at least 3 or 4 orders of magintude better results
criterion = nn.MSELoss() 
#criterion = YOLOLoss().to("cuda")

#Adam converges faster than SGD
#SGD usage might me safer in general
#Both work well if weitht_decay term (added to loss function) from interval [0.005;0.001]
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

#number of epoch should be around 50 -  increasing it does not provide with muuch additional gain
num_epochs = 50
#num_epochs = 100

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    model.train()
    epoch_loss = 0
    for images, labels in dataset:
        images, labels = images.to("cuda"), labels.to("cuda")
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
        images, labels = images.to("cuda"), labels.to("cuda")
        predictions = model(images)
        loss = criterion(predictions, labels)
        test_loss += loss.item()
print(f"Test Loss: {test_loss:.4f}")

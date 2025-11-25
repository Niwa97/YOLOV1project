import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

#Transforming VOC2012 classes to YOLO format - data set with 20 classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
#Paths set for google collab environment
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

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

#Transforms images data to YOLOv1 output
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, S=7, C=20):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        self.S = S
        self.C = C

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
        labels = torch.zeros((self.S, self.S, 5 + self.C))

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    grid_x = int(x_center * self.S)
                    grid_y = int(y_center * self.S)
         
                    grid_x = min(self.S - 1, max(0, grid_x))
                    grid_y = min(self.S - 1, max(0, grid_y))

                    labels[grid_y, grid_x, 0] = x_center
                    labels[grid_y, grid_x, 1] = y_center
                    labels[grid_y, grid_x, 2] = width
                    labels[grid_y, grid_x, 3] = height
                    labels[grid_y, grid_x, 4] = 1.0
                    labels[grid_y, grid_x, 5 + class_id] = 1.0 

        return image, labels

#Loss function prepared for YOLO models
#In v1/v2 version loss function is fairly simple - new versions uses much more sophisticated approach
#We sum three main loss terms:
#Localization loss - Control errors in box predictions, penalizes bad localization of center of cells and inaccurate height and width.
#Confidence loss - Controllss the confidence of the most accurate box and tries to make confidence score close to 0
#Classification loss - Penalize incorrect class predictions

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        N = predictions.shape[0]
        device = predictions.device
        preds = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        pred_boxes = preds[..., : self.B * 5].view(-1, self.S, self.S, self.B, 5)
        pred_cls = preds[..., self.B * 5:] 

        tgt_box = targets[..., :5]
        tgt_cls = targets[..., 5:]

        obj_mask = tgt_box[..., 4] > 0
        noobj_mask = ~obj_mask
        ious = []
        for b in range(self.B):
            iou = self.compute_iou(pred_boxes[..., b, :4], tgt_box[..., :4])
            ious.append(iou)
        ious = torch.stack(ious, dim=-1)

        best_box_idx = torch.argmax(ious, dim=-1, keepdim=True)
        best_box_mask = torch.zeros_like(ious, dtype=torch.bool).scatter_(-1, best_box_idx, True)
        obj_mask_expanded = obj_mask.unsqueeze(-1).expand_as(best_box_mask)
        responsible_mask = best_box_mask & obj_mask_expanded
        resp_boxes = pred_boxes[responsible_mask].view(-1, 5)
        tgt_resp = tgt_box[obj_mask].view(-1, 5)

        if resp_boxes.numel() == 0:
            loc_loss = torch.tensor(0.0, device=device)
        else:
            loc_loss = self.lambda_coord * (
                torch.sum((resp_boxes[:, :2] - tgt_resp[:, :2]) ** 2) +
                torch.sum((torch.sqrt(resp_boxes[:, 2:4].clamp(min=1e-6)) -
                           torch.sqrt(tgt_resp[:, 2:4].clamp(min=1e-6))) ** 2)
            )

        conf_pred = pred_boxes[..., :, 4]
        conf_tgt = ious.detach() * obj_mask.unsqueeze(-1).float()

        conf_loss_obj = torch.sum((conf_pred[obj_mask.unsqueeze(-1).expand_as(conf_pred)] - conf_tgt[obj_mask.unsqueeze(-1).expand_as(conf_pred)]) ** 2)
        conf_loss_noobj = self.lambda_noobj * torch.sum((conf_pred[noobj_mask.unsqueeze(-1).expand_as(conf_pred)]) ** 2)
        
        if obj_mask.sum() == 0:
            cls_loss = torch.tensor(0.0, device=device)
        else:
            cls_loss = torch.sum((pred_cls[obj_mask] - tgt_cls[obj_mask]) ** 2)

        loss = loc_loss + conf_loss_obj + conf_loss_noobj + cls_loss
        return loss
        
# Calculating intersection over union metric
    def compute_iou(self, boxes1, boxes2):
        b1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
        b1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
        b1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
        b1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2

        b2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
        b2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
        b2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
        b2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - inter_area + 1e-9 #division by zero handling
        return inter_area / union 


def visualize_yolo_prediction(image, label, pred, class_names):

    image_np = image.permute(1,2,0).cpu().numpy()
    H, W = image_np.shape[:2]

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(image_np)
    ax.set_title("Actual (Green) vs Predicted (Red)")

    for gy in range(7):
        for gx in range(7):
            if label[gy, gx, 4] > 0.5:
                x, y, w, h = label[gy, gx, :4]
                cls = torch.argmax(label[gy, gx, 5:]).item()
                cls_name = class_names[cls]
                xc, yc = x * W, y * H
                bw, bh = w * W, h * H
                x0, y0 = xc - bw/2, yc - bh/2
                rect = patches.Rectangle((x0, y0), bw, bh, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                ax.text(x0, y0-5, cls_name, color='green')
                
        for gx in range(7):
            if pred[gy, gx, 4] > 0.3:
                x, y, w, h = pred[gy, gx, :4]
                cls = torch.argmax(pred[gy, gx, 5:]).item()
                cls_name = class_names[cls]
                conf = pred[gy, gx, 4].item()
                xc, yc = x * W, y * H
                bw, bh = w * W, h * H
                x0, y0 = xc - bw/2, yc - bh/2
                rect = patches.Rectangle((x0, y0), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x0, y0-5, f"{cls_name} ({conf:.2f})", color='red')
    plt.show()

#cuda ususally needed - cpu calculations are extremally slow
device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLOv1().to(device)

#Yolo loss should provide at least 3 or 4 orders of magintude better results than any standard error metric
criterion = YOLOLoss().to(device)

train_loader = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16, shuffle=True)
test_loader = DataLoader(YOLODataset(VOC_IMAGE_PATH, YOLO_LABELS_PATH, transform=transform), batch_size=16, shuffle=True)

#Adam converges faster than SGD
#SGD usage might me safer in general
#Both work well if weitht_decay term (added to loss function) from interval [0.005;0.001]
#Learing rate at most 10^(-4) (around 10^(-6) ideally)- otherwise we can expect numerical errors
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)

#number of epoch should be around 60/70 -  increasing it does not provide with muuch additional gain
num_epochs = 70

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0.0
    for images, labels in tqdm(train_loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Train Loss: {epoch_loss:.4f}")

model.eval()
test_loss = 0.0
shown = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="test"):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        predictions = torch.sigmoid(predictions) # lowering LR for stability
        predictions = predictions.clamp(1e-6, 1-1e-6)
        loss = criterion(predictions, labels)
        test_loss += loss.item()

        if shown < 6:
            idx = random.randint(0, images.size(0)-1)

            visualize_yolo_prediction(
                image=images[idx].cpu(),
                label=labels[idx].cpu(),
                pred=predictions[idx].cpu(),
                class_names=VOC_CLASSES
            )
            shown += 1
            
print(f"Test Loss: {test_loss:.4f}")





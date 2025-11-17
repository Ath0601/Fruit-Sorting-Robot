import os
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision.transforms.functional as F
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Custom Compose for (image, target) pair
# ============================================================
class ComposeWithBoxes:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# ============================================================
# Custom Transforms
# ============================================================
class ResizeWithBoxes:
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, target):
        w, h = image.size
        new_w, new_h = self.size
        scale_x, scale_y = new_w / w, new_h / h

        image = F.resize(image, self.size)
        boxes = target["boxes"]
        boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        target["boxes"] = boxes
        return image, target


class ToTensorWithBoxes:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class NormalizeWithBoxes:
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train=True):
    return ComposeWithBoxes([
        ResizeWithBoxes((300, 300)),
        ToTensorWithBoxes(),
        NormalizeWithBoxes(),
    ])

# ============================================================
# Dataset
# ============================================================
class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = ['background', 'apple', 'banana', 'orange', 'mixed']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_files = sorted(list(self.root_dir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images in {split} split")

    def __len__(self):
        return len(self.image_files)

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx.get(name, 0))
        return boxes, labels

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        xml_path = img_path.with_suffix(".xml")

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_xml(xml_path)

        if len(boxes) == 0:
            boxes = [[0, 0, 10, 10]]
            labels = [0]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transform:
            image, target = self.transform(image, target)
        return image, target

# ============================================================
# Model creation
# ============================================================
def create_ssd_model(num_classes):
    model = ssd300_vgg16(weights="DEFAULT")
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes
    )
    return model

# ============================================================
# Loss function
# ============================================================
def calculate_total_loss(loss_output):
    if isinstance(loss_output, list):
        return sum(calculate_total_loss(lo) for lo in loss_output)
    if isinstance(loss_output, dict):
        return sum(loss for loss in loss_output.values())
    if isinstance(loss_output, torch.Tensor):
        return loss_output
    return torch.tensor(loss_output, dtype=torch.float32)

# ============================================================
# Training function
# ============================================================
def train_model(model, train_loader, val_loader, device, epochs=50, resume_checkpoint=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0
    train_losses, val_losses = [], []

    # Resume if checkpoint exists
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        print(f"✅ Resumed at epoch {start_epoch}")

    print("🚀 Starting training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = calculate_total_loss(loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_train_loss += losses.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i}/{len(train_loader)}, Loss: {losses.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.train()  # Keep model in training mode for loss computation
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss_value = calculate_total_loss(loss_dict)

                if isinstance(loss_value, torch.Tensor):
                    total_val_loss += loss_value.item()
                else:
                    total_val_loss += float(loss_value)

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses
            }
            ckpt_path = f"ssd_fruit_checkpoint_epoch_{epoch+1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    return train_losses, val_losses

# ============================================================
# Plot losses
# ============================================================
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("training_loss_curve.png")
    print("✅ Saved training curve as training_loss_curve.png")

# ============================================================
# Main
# ============================================================
def main():
    data_path = "/home/atharva/quanser_ws/merged_fruit_detection_dataset"
    num_classes = 5
    batch_size = 8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"🎉 Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ Using CPU")

    train_dataset = FruitDataset(data_path, transform=get_transform(True), split="train")
    val_dataset = FruitDataset(data_path, transform=get_transform(False), split="valid")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)

    model = create_ssd_model(num_classes)
    print("✅ Model created successfully")

    start = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader, device,
                                           epochs=epochs, resume_checkpoint=None)
    end = time.time()

    print(f"\n🏁 Training completed in {(end - start)/60:.2f} minutes.")
    plot_losses(train_losses, val_losses)

    final_path = "ssd_fruit_detector_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved as {final_path}")


if __name__ == "__main__":
    main()

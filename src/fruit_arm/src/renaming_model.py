import torch
from torchvision.models.detection import ssd300_vgg16

# === Step 1: Define your SSD architecture ===
# Adjust num_classes based on your dataset (include background)
NUM_CLASSES = 5  # e.g., background + 3 fruits

model = ssd300_vgg16(weights=None, num_classes=NUM_CLASSES)

# === Step 2: Load checkpoint safely ===
checkpoint_path = "/home/atharva/quanser_ws/ssd_fruit_checkpoint_best.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Handle different checkpoint formats
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint  # plain state_dict
model.load_state_dict(state_dict, strict=False)

model.eval()

# === Step 3: Prepare dummy input for tracing ===
example_input = torch.randn(1, 3, 300, 300)

# === Step 4: Convert to TorchScript ===
try:
    scripted_model = torch.jit.script(model)
except Exception as e:
    print("Scripting failed, using tracing instead:", e)
    scripted_model = torch.jit.trace(model, example_input)

# === Step 5: Save the TorchScript model ===
output_path = "/home/atharva/quanser_ws/ssd_fruit_best.pt"
scripted_model.save(output_path)

print(f"TorchScript model saved at: {output_path}")

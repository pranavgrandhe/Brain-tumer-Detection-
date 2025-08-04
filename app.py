# === FIX SSL ERROR BEFORE IMPORTING GRADIO ===
import os
os.environ.pop("SSL_CERT_FILE", None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

# SE Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
    def forward(self, x):
        b, c, _, _ = x.size()
        s = x.view(b, c, -1).mean(dim=2)
        e = F.relu(self.fc1(s))
        e = torch.sigmoid(self.fc2(e)).view(b, c, 1, 1)
        return x * e

# ResNet-Hybrid
class CNN_Transformer_Hybrid(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.se = SEBlock(2048)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.token = nn.Linear(2048, 512)
        self.drop1 = nn.Dropout(0.3)
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        f = self.cnn(x)
        f = self.se(f)
        f = self.pool(f).view(f.size(0), -1)
        t = self.drop1(self.token(f)).unsqueeze(1)
        t = self.transformer(t).mean(dim=1)
        return self.fc(self.drop2(t))

# EfficientNet-Hybrid
class EfficientNet_Transformer_Hybrid(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        eff = models.efficientnet_b0(pretrained=False)
        self.cnn = eff.features
        self.se = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.token = nn.Linear(1280, 512)
        self.drop1 = nn.Dropout(0.3)
        enc = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, num_layers=4)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        f = self.cnn(x)
        f = self.se(f)
        f = self.pool(f).view(f.size(0), -1)
        t = self.drop1(self.token(f)).unsqueeze(1)
        t = self.transformer(t).mean(dim=1)
        return self.fc(self.drop2(t))

# Load models
checkpoint = torch.load("hybrid_models.pth", map_location=device)

resnet_model = CNN_Transformer_Hybrid().to(device)
resnet_model.load_state_dict(checkpoint["resnet_hybrid"])
resnet_model.eval()

effnet_model = EfficientNet_Transformer_Hybrid().to(device)
effnet_model.load_state_dict(checkpoint["effnet_hybrid"])
effnet_model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: torch.clamp(x + 0.02 * torch.randn_like(x), -1.0, 1.0)),
])

# Grad-CAM 
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.grad, self.act = None, None
        self.h1 = self.layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o.detach()))
        self.h2 = self.layer.register_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0].detach()))
    def remove_hooks(self):
        self.h1.remove()
        self.h2.remove()
    def __call__(self, x, class_idx=None):
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        out.backward(gradient=one_hot, retain_graph=True)
        w = self.grad.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)
        return cam.cpu().numpy()[0,0], class_idx

# Saliency 
def compute_saliency(model, x, class_idx=None):
    x.requires_grad_()
    out = model(x)
    if class_idx is None:
        class_idx = out.argmax(dim=1).item()
    val = out[0, class_idx]
    model.zero_grad()
    val.backward()
    sal, _ = torch.max(x.grad.abs(), dim=1)
    sal = (sal - sal.min())/(sal.max()-sal.min()+1e-8)
    return sal.squeeze().cpu().numpy()

# Gradio prediction function
def predict(img: Image.Image):
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    img_resized = np.array(img.resize((224, 224)))

    with torch.no_grad():
        out1 = resnet_model(img_tensor)
        out2 = effnet_model(img_tensor)
        final_out = (out1 + out2) / 2
        probs = F.softmax(final_out, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    # Grad-CAM for ResNet
    cam_r = GradCAM(resnet_model, resnet_model.cnn[-1])
    cam_map_r, _ = cam_r(img_tensor, pred_idx)
    cam_r.remove_hooks()
    cam_overlay_r = np.uint8(255 * cam_map_r)
    cam_overlay_r = cv2.applyColorMap(cam_overlay_r, cv2.COLORMAP_JET)
    cam_result_r = cv2.addWeighted(img_resized, 0.5, cam_overlay_r, 0.5, 0)

    # Grad-CAM for EfficientNet
    cam_e = GradCAM(effnet_model, effnet_model.cnn[-1])
    cam_map_e, _ = cam_e(img_tensor, pred_idx)
    cam_e.remove_hooks()
    cam_overlay_e = np.uint8(255 * cam_map_e)
    cam_overlay_e = cv2.applyColorMap(cam_overlay_e, cv2.COLORMAP_JET)
    cam_result_e = cv2.addWeighted(img_resized, 0.5, cam_overlay_e, 0.5, 0)

    # Saliency for ResNet
    sal_r = compute_saliency(resnet_model, img_tensor.clone(), pred_idx)
    sal_r = np.uint8(255 * sal_r)
    sal_r = cv2.applyColorMap(sal_r, cv2.COLORMAP_HOT)

    # Saliency for EfficientNet
    sal_e = compute_saliency(effnet_model, img_tensor.clone(), pred_idx)
    sal_e = np.uint8(255 * sal_e)
    sal_e = cv2.applyColorMap(sal_e, cv2.COLORMAP_HOT)

    return (
        {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        cam_result_r,
        cam_result_e,
        sal_r,
        sal_e,
    )

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=4, label="Predicted Probabilities"),
        gr.Image(type="numpy", label="Grad-CAM (ResNet)"),
        gr.Image(type="numpy", label="Grad-CAM (EffNet)"),
        gr.Image(type="numpy", label="Saliency Map (ResNet)"),
        gr.Image(type="numpy", label="Saliency Map (EffNet)")
    ],
    title="Brain Tumor Classifier",
    description="Ensemble of ResNet-Hybrid and EffNet-Hybrid. Shows Grad-CAM and Saliency visualizations from both models."
)

if __name__ == "__main__":
    iface.launch()

import os
import cv2
import torch
import timm
import gdown
import numpy as np
import gradio as gr
import torchvision.transforms as transforms
from torch import nn

# ==============================
# إعدادات الموديل
# ==============================
MODEL_PATH = "best_vit_lstm.pt"
MODEL_ID = "1GjmrQSLRtCwAtkk30ZOtFFXFqhOg6BxX"  # <-- ID من Google Drive

# تحميل الموديل لو مش موجود
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# تعريف الموديل
# ==============================
class ViT_LSTM_Classifier(nn.Module):
    def __init__(self, vit_name="vit_tiny_patch16_224", lstm_hidden=256, lstm_layers=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0)
        self.feat_dim = self.vit.num_features if hasattr(self.vit, "num_features") else 192
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits

# ==============================
# تحميل الموديل المدرب
# ==============================
model = ViT_LSTM_Classifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==============================
# الإعدادات المسبقة للفريمات
# ==============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

SEQ_LEN = 8
frames_buffer = []

# ==============================
# دالة التنبؤ
# ==============================
def detect(frame):
    global frames_buffer

    if frame is None:
        return None, "No frame detected"
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames_buffer.append(transform(frame))
    
    if len(frames_buffer) > SEQ_LEN:
        frames_buffer.pop(0)

    if len(frames_buffer) == SEQ_LEN:
        clip = torch.stack(frames_buffer).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(clip)
            pred = torch.argmax(output, dim=1).item()
        label = "🚨 Violent Behavior Detected!" if pred == 1 else "✅ Normal Behavior"
        color = (255, 0, 0) if pred == 1 else (0, 255, 0)
    else:
        label = "Collecting frames..."
        color = (255, 255, 0)
    
    frame_disp = cv2.putText(frame.copy(), label, (30, 50),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return frame_disp, label

# ==============================
# واجهة Gradio
# ==============================
css = """
#alert-audio { display: none; }
"""

js_alert = """
function playSound(label){
  const audio = document.getElementById("alert-audio");
  if(label.includes("Violent")) audio.play();
}
"""

with gr.Blocks(js=js_alert, css=css, title="Violence Detection (ViT + LSTM)") as demo:
    gr.HTML("<h1 style='text-align:center; color:#d32f2f;'>Violence Detection System</h1>")
    gr.HTML("<audio id='alert-audio' src='file/alert.wav'></audio>")

    webcam = gr.Image(streaming=True, label="Camera Feed", sources=["webcam"])
    video_output = gr.Image(label="Live Detection Output")
    text_output = gr.Textbox(label="Prediction", interactive=False)

    webcam.stream(fn=detect, inputs=webcam, outputs=[video_output, text_output])
    gr.HTML("<script>setInterval(() => { playSound(document.querySelector('textarea').value); }, 500);</script>")

demo.launch(server_name="0.0.0.0", server_port=7860)

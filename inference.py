# === Forgery localization (Spatial + Temporal) with confidence overlay (Top-left) ===
# Paste this cell AFTER your Model class definition.
# Expects best_model.pth in notebook working directory.
# Requires: torch, torchvision, numpy, cv2, matplotlib, tqdm
import os
import gdown

MODEL_PATH = "last_epoch_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

from model import Model

import os, cv2, copy, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import torch
torch.backends.cudnn.enabled = False


#Model with feature visualization

SEQ_LEN = 20
FRAME_SIZE = (112, 112)        # (W, H)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_OUTPUT = "forgery_localized_output_with_confidence.mp4"
TEMPORAL_BORDER_THRESHOLD = 0.4
OCCLUSION_MODE = True
# ----------------------------

# ---------- Utilities ----------
def select_video_via_dialog():
    print("Select video file...")
    path = askopenfilename(title="Select a video")
    if path == '':
        raise FileNotFoundError("No file selected.")
    return path

def extract_frames(video_path, seq_len=SEQ_LEN, resize=FRAME_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames extracted from video.")
    while len(frames) < seq_len:
        frames.append(frames[-1].copy())
    frames = np.stack(frames[:seq_len], axis=0)
    return frames

def preprocess_frames(frames):
    arr = frames.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,1,3)
    arr = (arr - mean) / std
    t = torch.tensor(arr).permute(0,3,1,2).unsqueeze(0)  # (1, seq_len, 3, H, W)
    return t.float()

def overlay_heatmap_rgb(frame_rgb, cam_map, alpha=0.45):
    heat = cv2.applyColorMap((cam_map*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (frame_rgb.astype(np.float32)*(1-alpha) + heat.astype(np.float32)*alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def draw_border_and_label(frame_rgb, label_text, color=(255,0,0), thickness=4):
    h,w = frame_rgb.shape[:2]
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (0,0), (w-1,h-1), bgr, thickness)
    # put label top-left, small
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_bgr, label_text, (8, 26), font, 0.7, bgr, 2, cv2.LINE_AA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# ---------- Hooks & Grad-CAM ----------
class ActGrad:
    def __init__(self):
        self.activations = None
        self.gradients = None
    def forward_hook(self, module, inp, out):
        self.activations = out
    def backward_hook(self, grad):
        self.gradients = grad

def register_forward_hook(model):
    ag = ActGrad()
    handle = model.model.register_forward_hook(lambda m, i, o: ag.forward_hook(m,i,o))
    return ag, handle

def compute_gradcam_for_sequence(model, input_tensor, class_idx=None):
    model.zero_grad()
    ag, handle = register_forward_hook(model)
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad = True
    fmap, logits = model(input_tensor)  # fmap: (batch*seq_len, C, Hf, Wf)
    if class_idx is None:
        class_idx = torch.argmax(logits, dim=1).item()
    score = logits[0, class_idx]
    score.backward(retain_graph=True)
    # fetch activations and gradients
    acts = ag.activations if ag.activations is not None else fmap.detach()
    grads = None
    # sometimes grads saved on fmap.grad
    if hasattr(fmap, "grad") and fmap.grad is not None:
        grads = fmap.grad.detach()
    else:
        # try to get gradients by autograd on acts (if available)
        try:
            grads = torch.autograd.grad(score, acts, retain_graph=True, allow_unused=True)[0]
        except Exception:
            pass
    if grads is None:
        raise RuntimeError("Unable to capture gradients for Grad-CAM. Ensure model & DEVICE match and input requires_grad.")
    acts_np = acts.detach().cpu().numpy()
    grads_np = grads.detach().cpu().numpy()
    weights = np.mean(grads_np, axis=(2,3))  # (batch*seq_len, C)
    cams = []
    for i in range(acts_np.shape[0]):
        cam = np.zeros((acts_np.shape[2], acts_np.shape[3]), dtype=np.float32)
        for c in range(acts_np.shape[1]):
            cam += weights[i, c] * acts_np[i, c]
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cams.append(cam)
    cams = np.stack(cams, axis=0)
    cams_resized = np.zeros((cams.shape[0], FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.float32)
    for i in range(cams.shape[0]):
        cams_resized[i] = cv2.resize(cams[i], FRAME_SIZE)
    handle.remove()
    return cams_resized

# ---------- Temporal attribution ----------
def temporal_gradients(model, input_tensor, class_idx=None):
    model.zero_grad()
    ag, handle = register_forward_hook(model)
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad = True
    fmap, logits = model(input_tensor)
    if class_idx is None:
        class_idx = torch.argmax(logits, dim=1).item()
    score = logits[0, class_idx]
    score.backward(retain_graph=True)
    acts = ag.activations
    grads = None
    # try to get grads directly
    if acts is None:
        raise RuntimeError("Could not capture activations for temporal gradient.")
    grads = acts.grad if hasattr(acts, "grad") and acts.grad is not None else None
    if grads is None:
        try:
            grads = torch.autograd.grad(score, acts, retain_graph=True, allow_unused=True)[0]
        except Exception:
            pass
    if grads is None:
        raise RuntimeError("Unable to compute temporal gradients.")
    pooled = F.adaptive_avg_pool2d(acts, (1,1)).squeeze(-1).squeeze(-1)   # (batch*seq_len, C)
    grads_pooled = torch.mean(grads, dim=(2,3))
    imp = torch.norm(grads_pooled, dim=1).cpu().numpy()
    seq_len = input_tensor.shape[1]
    if imp.size != seq_len:
        imp = imp[:seq_len]
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
    handle.remove()
    return imp

def temporal_occlusion_importance(model, input_tensor, class_idx=None):
    model.eval()
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        _, logits_orig = model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(logits_orig, dim=1).item()
        base_score = F.softmax(logits_orig, dim=1)[0, class_idx].item()
    seq_len = input_tensor.shape[1]
    drops = []
    for t in range(seq_len):
        modified = input_tensor.clone()
        modified[0, t] = 0.0
        with torch.no_grad():
            _, logits = model(modified)
            score = F.softmax(logits, dim=1)[0, class_idx].item()
        drops.append(max(0.0, base_score - score))
    drops = np.array(drops)
    if drops.max() > 0:
        drops = drops / (drops.max() + 1e-8)
    return drops

def combined_temporal_score(grad_imp, occ_imp, alpha=0.7):
    if occ_imp is None:
        return grad_imp
    assert len(grad_imp) == len(occ_imp)
    combined = alpha * grad_imp + (1 - alpha) * occ_imp
    if combined.max() > 0:
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    return combined

# ---------- Main pipeline ----------
def run_forgery_localization_pipeline(video_path, weights_path="last_epoch_model.pth"):
    print("Loading trained model...")
    model = Model(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("last_epoch_model.pth", map_location=DEVICE))
    model.eval()
    print("Loaded:", weights_path)

    print("Extracting frames...")
    frames = extract_frames(video_path, seq_len=SEQ_LEN, resize=FRAME_SIZE)  # (seq_len, H, W, 3)
    input_tensor = preprocess_frames(frames).to(DEVICE)  # (1, seq_len, 3, H, W)

    # forward once for prediction & confidence
    with torch.no_grad():
     fmap, logits = model(input_tensor)
     probs = F.softmax(logits, dim=1)[0]

# FIX 2: Avoid NaN
    if torch.isnan(probs).any():
        print("Warning: NaN in probabilities → forcing zeros")
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

    # Always move to CPU + numpy AFTER fixing NaN
    probs = probs.cpu().numpy()

    # Compute prediction & confidence
    pred_class = int(np.argmax(probs))
    label_text = "FAKE" if pred_class == 0 else "REAL"
    conf = probs[pred_class]  # in [0,1]

    print("Prediction:", label_text, "Confidence:", conf)


    # Spatial Grad-CAM
    try:
        cams = compute_gradcam_for_sequence(model, input_tensor, class_idx=None)
    except Exception as e:
        print("Grad-CAM failed:", e)
        cams = np.zeros((SEQ_LEN, FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.float32)

    # Temporal
    try:
        grad_imp = temporal_gradients(model, input_tensor, class_idx=None)
    except Exception as e:
        print("Temporal gradient failed:", e)
        grad_imp = np.zeros(SEQ_LEN, dtype=np.float32)

    occ_imp = None
    if OCCLUSION_MODE:
        try:
            occ_imp = temporal_occlusion_importance(model, input_tensor, class_idx=None)
        except Exception as e:
            print("Occlusion failed:", e)
            occ_imp = None

    combined = combined_temporal_score(grad_imp, occ_imp, alpha=0.7)
    suspicious_flags = combined >= TEMPORAL_BORDER_THRESHOLD

    # Create annotated frames with confidence (top-left)
    annotated_frames = []
    for i in range(SEQ_LEN):
        orig = frames[i].copy()  # RGB uint8
        cam = cams[i] if i < cams.shape[0] else np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.float32)
        overlay = overlay_heatmap_rgb(orig, cam, alpha=0.45)

        # Add small bottom bar showing temporal map
        bar_h = 8
        overlay_padded = np.pad(overlay, ((0, bar_h),(0,0),(0,0)), mode='constant', constant_values=0)
        W = overlay_padded.shape[1]
        for t_idx in range(SEQ_LEN):
            x0 = int(W * (t_idx / SEQ_LEN))
            x1 = int(W * ((t_idx + 1)/SEQ_LEN))
            val = combined[t_idx]
            if val < 0.33:
                col = (0,200,0)
            elif val < 0.66:
                col = (200,200,0)
            else:
                col = (200,0,0)
            cv2.rectangle(overlay_padded, (x0, overlay.shape[0]), (x1, overlay.shape[0]+bar_h), col, -1)
        # mark current frame index
        cur_x0 = int(W * (i/SEQ_LEN))
        cv2.rectangle(overlay_padded, (cur_x0, overlay.shape[0]), (cur_x0+2, overlay.shape[0]+bar_h), (255,255,255), -1)
        overlay = overlay_padded

        # Add confidence and labels at TOP-LEFT (location A)
        conf_text = f"{label_text} ({conf*100:.2f}%)"
        temp_text = f"TempScore: {combined[i]:.2f}"
        frame_text = f"Frame {i+1}/{SEQ_LEN}"

        # draw a semi-transparent rectangle background for readability
        h_txt = 70
        overlay[:h_txt, :260] = (overlay[:h_txt, :260] * 0.4 + 30)  # darken slightly
        # put texts (white)
        cv2.putText(overlay, conf_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, temp_text, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, frame_text, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)

        if suspicious_flags[i]:
            overlay = draw_border_and_label(overlay, "SUSPICIOUS", color=(255,0,0), thickness=6)

        annotated_frames.append(overlay)

    # Save video
    print("Saving suspicious frames only (no output video) ...")

    suspicious_count = 0

    # for i in range(SEQ_LEN):
    #   if suspicious_flags[i]:
    #     suspicious_count += 1
    #     save_path = f"suspicious_frame_{i+1}.png"
    #     cv2.imwrite(save_path, cv2.cvtColor(annotated_frames[i], cv2.COLOR_RGB2BGR))
    #     print(f"Saved: {save_path}")

    # if suspicious_count == 0:
    #   print("No suspicious frames detected.")
    # else:
    #   print(f"Total suspicious frames saved: {suspicious_count}")

    # Plot combined temporal importance inline
    try:
        plt.figure(figsize=(10,2))
        plt.plot(combined, marker='o')
        plt.title("Combined temporal importance (frame index)")
        plt.xlabel("Frame index")
        plt.ylabel("importance (0-1)")
        plt.grid(True)
        plt.show()
    except Exception:
        pass

    return {
        "prediction_class": pred_class,
        "probabilities": probs,
        "confidence": conf,
        "gradcam_maps": cams,
        "grad_imp": grad_imp,
        "occ_imp": occ_imp,
        "combined_temporal": combined,
        "suspicious_flags": suspicious_flags,
        "output_path": VIDEO_OUTPUT
    }

# === Run the pipeline ===
# from google.colab import files

# print("Upload your video (.mp4, .avi, etc.)")
# uploaded = files.upload()

# video_path = "./video1.mp4"
# print("Using video:", video_path)
# results = run_forgery_localization_pipeline(video_path)
# print("Prediction:", "FAKE" if results["prediction_class"]==0 else "REAL", f"({results['confidence']*100:.2f}%)")

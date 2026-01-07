# from fer import FER

# import cv2
# import torch
# import numpy as np
# from torchvision import transforms
# from PIL import Image

# from camera.webcam import Webcam
# from engine.style_engine import StyleEngine

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from models.fast_style.transformer_net import TransformerNet




# # model = TransformerNet()
# ##

# styles = {
#     '1': 'mosaic.pth',
#     '2': 'candy.pth',
#     '3': 'udnie.pth'
# }

# current_style = '3'

# def load_style(style_key):
#     model = TransformerNet()
#     sd = torch.load(f"models/fast_style/{styles[style_key]}", map_location=device)
#     model.load_state_dict(sd)
#     return model.to(device).eval()


# model = load_style(current_style)
# engine = StyleEngine(model, device)

# #########

# # state_dict = torch.load("models/fast_style/mosaic.pth", map_location=device)
# # model.load_state_dict(state_dict)
# # model = model.to(device).eval()


# engine = StyleEngine(model, device)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.mul(255))
# ])

# cam = Webcam(width=640, height=480)


# ######
# alpha = 0.7  # style strength

# ######

# while True:
#     frame = cam.read()
#     ##3
#     frame = cv2.resize(frame, (512, 512))

#     ##
#     if frame is None:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(rgb)

#     tensor = transform(img).unsqueeze(0).to(device)

#     output = engine.stylize(tensor)
#     output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
#     output = np.clip(output, 0, 255).astype(np.uint8)

#     # cv2.imshow("Style Time Machine", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

#     ##3
#     final = cv2.addWeighted(
#         cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
#         alpha,
#         frame,
#         1 - alpha,
#         0
#     )

#     cv2.imshow("Style Time Machine", final)

#     #####
    
#     ###3
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('+'):
#         alpha = min(alpha + 0.05, 1.0)

#     elif key == ord('-'):
#         alpha = max(alpha - 0.05, 0.0)

#     elif chr(key) in styles:
#         model = load_style(chr(key))
#         engine = StyleEngine(model, device)

#     elif key == ord('q'):
#         break

#     ###
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# cam.release()
# cv2.destroyAllWindows()




# ===============================
# IMPORTS
# ===============================
from utils.diffusion_snapshot import generate_snapshot
import os
from datetime import datetime


import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from fer import FER

from camera.webcam import Webcam
from engine.style_engine import StyleEngine
from models.fast_style.transformer_net import TransformerNet


DIFFUSION_PROMPTS = {
    "anime": (
       "anime portrait, clean lineart, sharp expressive eyes, "
    "accurate face proportions, smooth shading, "
    "studio ghibli inspired, high quality anime illustration, "
    "clear contours, no distortion"
),

    "cartoon": (
       "clean cartoon portrait, bold smooth outlines, "
    "simple but accurate facial proportions, "
    "flat colors, cel shading, sharp edges, "
    "pixar style illustration, 2D digital art, "
    "high clarity, no blur")
}

current_diffusion_style = "anime"

OUTPUT_DIR = "generated_art"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# DEVICE & PERFORMANCE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ===============================
# STYLE CONFIG
# ===============================
styles = {
    '1': 'mosaic.pth',   # structured
    '2': 'candy.pth',    # painterly
    '3': 'udnie.pth'     # sketch-like
}

current_style = '3'

def load_style(style_key):
    model = TransformerNet()
    sd = torch.load(f"models/fast_style/{styles[style_key]}", map_location=device)
    model.load_state_dict(sd)
    return model.to(device).eval()

model = load_style(current_style)
engine = StyleEngine(model, device)

# ===============================
# TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# ===============================
# CAMERA
# ===============================
cam = Webcam(width=640, height=480)

# ===============================
# EMOTION DETECTOR
# ===============================
emotion_detector = FER(mtcnn=False)
current_emotion = "neutral"
emotion_counter = 0

# ===============================
# MOTION VARIABLES
# ===============================
prev_gray = None
motion_strength = 0.0

# ===============================
# STYLE CONTROL
# ===============================
alpha = 0.6  # default style strength

# ===============================
# FULLSCREEN WINDOW
# ===============================
cv2.namedWindow("Style Time Machine", cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty(
    "Style Time Machine",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ===============================
# MAIN LOOP
# ===============================
while True:
    frame = cam.read()
    if frame is None:
        break

    # Resize for performance
    frame = cv2.resize(frame, (512, 512))

    # ===============================
    # MOTION ANALYSIS (Option A)
    # ===============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_strength = np.mean(diff)
    prev_gray = gray

    motion_norm = min(motion_strength / 25.0, 1.0)
    alpha = 0.8 - (motion_norm * 0.5)
    alpha = np.clip(alpha, 0.3, 0.8)

    # ===============================
    # EMOTION ANALYSIS (Option B)
    # ===============================
    emotion_counter += 1
    if emotion_counter % 15 == 0:
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            current_emotion = max(
                emotions[0]["emotions"],
                key=emotions[0]["emotions"].get
            )

    # Emotion → Style mapping
    if current_emotion in ["happy", "surprise"]:
        target_style = '2'
    elif current_emotion in ["neutral", "sad"]:
        target_style = '1'
    else:
        target_style = '3'

    if target_style != current_style:
        current_style = target_style
        model = load_style(current_style)
        engine = StyleEngine(model, device)

    # ===============================
    # STYLE TRANSFER
    # ===============================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    tensor = transform(img).unsqueeze(0).to(device)

    try:
        output = engine.stylize(tensor)
        output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        output = np.clip(output, 0, 255).astype(np.uint8)
    except Exception as e:
        print("Inference error:", e)
        output = frame.copy()

    # ===============================
    # BLEND ORIGINAL + STYLED
    # ===============================
    final = cv2.addWeighted(
        cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
        alpha,
        frame,
        1 - alpha,
        0
    )

    cv2.imshow("Style Time Machine", final)

    # ===============================
    # KEY CONTROLS
    # ===============================
    # ===============================
    # KEY CONTROLS
    # ===============================
    key = cv2.waitKey(1) & 0xFF

    # Increase / decrease style strength
    if key == ord('+'):
        alpha = min(alpha + 0.05, 1.0)

    elif key == ord('-'):
        alpha = max(alpha - 0.05, 0.0)

    # Switch FAST style models (live NST)
    elif chr(key) in styles:
        current_style = chr(key)
        model = load_style(current_style)
        engine = StyleEngine(model, device)

    # -------------------------------
    # DIFFUSION SNAPSHOT MODE
    # -------------------------------
    elif key == ord('s'):
    # ===============================
    # SHOW GENERATING MESSAGE
    # ===============================
        overlay = final.copy()
        cv2.rectangle(overlay, (0, 0), (512, 80), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            "Generating Art... (Cloud GPU)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        cv2.imshow("Style Time Machine", overlay)
        cv2.waitKey(1)

        # ===============================
        # SAVE SNAPSHOT LOCALLY
        # ===============================
        cv2.imwrite("snapshot/input.png", frame)
        print("Snapshot saved to snapshot/input.png")

        # ===============================
        # TRIGGER KAGGLE DIFFUSION
        # ===============================
        from remote_diffusion import run_kaggle_diffusion

        try:
            run_kaggle_diffusion(
         DIFFUSION_PROMPTS[current_diffusion_style]
            )
        except Exception as e:
            print("Kaggle step failed:", e)
            print("Returning to live mode.")

        # ===============================
        # LOAD RESULT FROM KAGGLE
        # ===============================
        try:
            art = Image.open("results/output.png")

            art_np = np.array(art.convert("RGB"), dtype=np.uint8)
            cv2.imshow(
                "Style Time Machine",
                cv2.cvtColor(art_np, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(0)

        except Exception as e:
            print("Failed to load diffusion result:", e)
            print("Returning to live mode.")

        

        # # Show diffusion result (gallery-style pause)
        # art_np = np.array(art)
        # cv2.imshow(
        #     "Style Time Machine",
        #     cv2.cvtColor(art_np, cv2.COLOR_RGB2BGR)
        # )
        # cv2.waitKey(0)  # wait until any key

    # Switch DIFFUSION style
    elif key == ord('a'):
        current_diffusion_style = "anime"

    elif key == ord('c'):
        current_diffusion_style = "cartoon"

    # Quit safely
    elif key == ord('q'):
        break
    # break

# ===============================
# CLEANUP
# ===============================
cam.release()
cv2.destroyAllWindows()

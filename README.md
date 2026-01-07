# 🎨 Style-Transfer

**Real-Time Neural Style Transfer with Adaptive Processing & Diffusion Snapshots**

Style-Transfer is a real-time artistic video processing system that applies **neural style transfer** to a live webcam feed while dynamically adapting style strength and selection based on **motion** and **facial emotion**.
The system also supports **high-quality artistic snapshot generation** using Stable Diffusion (local or cloud-based).

This project combines **computer vision, deep learning, and interactive systems** into a single end-to-end application.

---

## ✨ Key Features

* **Real-Time Neural Style Transfer**

  * Live webcam stylization at interactive frame rates
  * Uses pre-trained fast style transfer models (TransformerNet)

* **Adaptive Style Control**

  * Motion-aware blending automatically adjusts style intensity
  * Emotion-aware style switching using facial expression recognition

* **Multiple Artistic Styles**

  * Mosaic (structured)
  * Candy (painterly)
  * Udnie (sketch-like)

* **High-Quality Snapshot Generation**

  * Capture a live frame and generate refined artwork
  * Supports:

    * Manual cloud GPU workflow (Kaggle)
    * Local Stable Diffusion (optional)

* **GPU-Accelerated Pipeline**

  * CUDA + cuDNN optimizations
  * Automatic device selection (CPU / GPU)

---

## 🧠 Why This Project?

Traditional neural style transfer demos are static and non-interactive.
This project explores **adaptive artistic intelligence**, where the system responds in real time to:

* **User motion**
* **User emotion**
* **Contextual interaction**

It demonstrates how deep learning models can be integrated into **real-time systems**, not just offline inference.

---

## 🏗️ System Architecture

### Real-Time Pipeline

1. Webcam frame capture
2. Motion analysis (frame differencing)
3. Emotion detection (FER)
4. Dynamic style selection & blending
5. GPU-accelerated neural style transfer
6. Fullscreen live visualization

### Snapshot Pipeline

1. User captures a frame
2. Snapshot saved locally
3. Diffusion generation triggered (cloud or local)
4. High-resolution artwork displayed

---


> ⚠️ Model weights are intentionally **not included** in the repository.

---

## 🚀 Installation

### Prerequisites

* Python 3.11+
* Webcam
* CUDA-compatible GPU (recommended)

### Setup

```bash
git clone https://github.com/iamnishaant/Style-Transfer.git
cd Style-Transfer
pip install -r requirements.txt
```

### Download Style Models

Place the following files inside `models/fast_style/`:

* `mosaic.pth`
* `candy.pth`
* `udnie.pth`

---

## ▶️ Running the Application

```bash
python main.py
```

The application starts in **fullscreen mode** using your webcam feed.

---

## 🎮 Controls

| Key         | Action                                |
| ----------- | ------------------------------------- |
| `1 / 2 / 3` | Switch style (Mosaic / Candy / Udnie) |
| `+ / -`     | Increase / decrease style intensity   |
| `s`         | Capture snapshot & generate art       |
| `a`         | Anime diffusion style                 |
| `c`         | Cartoon diffusion style               |
| `q`         | Quit                                  |

---

## ⚡ Performance Optimizations

* GPU inference with automatic device detection
* cuDNN benchmarking enabled
* Mixed-precision inference for diffusion
* Emotion detection throttling to reduce CPU load

---

## 🧪 Technologies Used

* **Deep Learning**: PyTorch, TorchVision
* **Computer Vision**: OpenCV
* **Emotion Recognition**: FER
* **Diffusion Models**: Hugging Face Diffusers
* **Acceleration**: CUDA, cuDNN

---

## 📌 Use Cases

* Interactive digital art installations
* AI-assisted creative tools
* Research on adaptive neural rendering
* Real-time human-AI interaction systems

---

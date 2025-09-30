#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from PIL import Image
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DeepfakeDetector:
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self, model_name: str = 'efficientnet_b4', device: Optional[str] = None):
        self.model_name = self._validate_model_name(model_name)
        self.device = self._setup_device(device)
        self.model = None
        self.transform = self._create_transforms()

        logger.info(f"Initialized DeepfakeDetector with {self.model_name} on {self.device}")

    def _validate_model_name(self, model_name: str) -> str:
        available_models = ['efficientnet_b4', 'efficientnet_b7', 'resnet50', 'vgg16']
        if model_name not in available_models:
            logger.warning(f"Model {model_name} not in recommended list. Available: {available_models}")
        return model_name

    def _setup_device(self, device: Optional[str]) -> str:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        if device == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        return device

    def _create_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self, weights_path: Optional[str] = None) -> None:
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=2
            )
            if weights_path:
                self._load_custom_weights(weights_path)
            else:
                self._setup_default_classifier()

            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _load_custom_weights(self, weights_path: str) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        logger.info(f"Loading custom weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    def _setup_default_classifier(self) -> None:
        logger.info("Setting up default binary classifier")
        if hasattr(self.model, 'classifier'):
            classifier_layer = 'classifier'
            in_features = self.model.classifier.in_features
        elif hasattr(self.model, 'fc'):
            classifier_layer = 'fc'
            in_features = self.model.fc.in_features
        elif hasattr(self.model, 'head'):
            classifier_layer = 'head'
            in_features = self.model.head.in_features
        else:
            raise RuntimeError("Could not find classifier layer in model")

        new_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )
        setattr(self.model, classifier_layer, new_classifier)

    def _validate_file_path(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _is_image_file(self, file_path: Union[str, Path]) -> bool:
        return Path(file_path).suffix.lower() in self.IMAGE_EXTENSIONS

    def _is_video_file(self, file_path: Union[str, Path]) -> bool:
        return Path(file_path).suffix.lower() in self.VIDEO_EXTENSIONS

    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict_image(self, image_path: str) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        file_path = self._validate_file_path(image_path)
        if not self._is_image_file(file_path):
            raise ValueError(f"Not a supported image format: {file_path.suffix}")
        logger.info(f"Analyzing image: {file_path.name}")
        image_tensor = self.preprocess_image(str(file_path))
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        predicted_class = predicted.item()
        prediction = "Real" if predicted_class == 0 else "Deepfake"
        return prediction, confidence.item()

    def predict_video(self, video_path: str, frame_interval: int = 10, max_frames: int = 100) -> Tuple[str, float]:
        file_path = self._validate_file_path(video_path)
        if not self._is_video_file(file_path):
            raise ValueError(f"Not a supported video format: {file_path.suffix}")
        logger.info(f"Analyzing video: {file_path.name}")
        cap = cv2.VideoCapture(str(file_path))
        frames, count = [], 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frames.append(frame)
            count += 1
        cap.release()
        deepfake_probs = []
        for frame in frames:
            frame_tensor = self.preprocess_image(frame)
            with torch.no_grad():
                outputs = self.model(frame_tensor)
                probs = torch.softmax(outputs, dim=1)
                deepfake_probs.append(probs[0][1].item())
        avg_prob = np.mean(deepfake_probs)
        prediction = "Deepfake" if avg_prob > 0.5 else "Real"
        confidence = avg_prob if prediction == "Deepfake" else (1 - avg_prob)
        return prediction, confidence

    def analyze_file(self, file_path: str, frame_interval: int = 10, max_frames: int = 100) -> dict:
        file_path_obj = self._validate_file_path(file_path)
        if self._is_image_file(file_path_obj):
            pred, conf = self.predict_image(str(file_path_obj))
            return {"file_type": "image", "prediction": pred, "confidence": conf, "success": True}
        elif self._is_video_file(file_path_obj):
            pred, conf = self.predict_video(str(file_path_obj), frame_interval, max_frames)
            return {"file_type": "video", "prediction": pred, "confidence": conf, "success": True}
        else:
            return {"success": False, "error": f"Unsupported format: {file_path_obj.suffix}"}


def create_argument_parser():
    parser = argparse.ArgumentParser(description="Deepfake Detection System")
    parser.add_argument("--file", "-f", type=str, help="Path to input file")
    parser.add_argument("--model", "-m", type=str, default="efficientnet_b4", help="Model architecture")
    parser.add_argument("--weights", "-w", type=str, help="Path to custom weights file")
    parser.add_argument("--device", "-d", type=str, choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Frame interval for videos")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames for videos")
    return parser


def cli_main():
    parser = create_argument_parser()
    args = parser.parse_args()
    detector = DeepfakeDetector(model_name=args.model, device=args.device)
    detector.load_model(weights_path=args.weights)
    if args.file:
        results = detector.analyze_file(args.file, frame_interval=args.interval, max_frames=args.max_frames)
        print(results)
    else:
        print("No file provided. Use GUI mode by running without arguments.")


def gui_main():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    detector = DeepfakeDetector()
    detector.load_model()

    def select_file():
        file_path = filedialog.askopenfilename(
            title="Select Image/Video",
            filetypes=[("Media files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm")]
        )
        if file_path:
            results = detector.analyze_file(file_path)
            output_text.delete(1.0, tk.END)
            if results["success"]:
                msg = f"{results['file_type'].upper()} Detected: {results['prediction']} ({results.get('confidence',0)*100:.2f}%)"
            else:
                msg = f"Error: {results['error']}"
            output_text.insert(tk.END, msg)
            messagebox.showinfo("Detection Result", msg)

    root = tk.Tk()
    root.title("Deepfake Detection System")
    root.geometry("600x400")

    tk.Label(root, text="Upload Image/Video for Deepfake Detection", font=("Arial", 14)).pack(pady=10)
    tk.Button(root, text="Select File", command=select_file, font=("Arial", 12), bg="lightblue").pack(pady=10)
    tk.Label(root, text="Detection Output:", font=("Arial", 12)).pack(pady=5)
    output_text = tk.Text(root, wrap="word", height=12, width=70, font=("Courier", 10))
    output_text.pack(padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        gui_main()

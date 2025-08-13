import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def show_mask_on_image(img: np.ndarray, mask: np.ndarray):
    if img.max()<=1: img = (img*255).astype("uint8")
    color = np.array([0,255,0],dtype=np.uint8)
    overlay = img.copy()
    overlay[mask>0] = (0.5*overlay[mask>0] + 0.5*color).astype("uint8")
    combined = cv2.addWeighted(img,0.6,overlay,0.4,0)
    plt.imshow(combined)
    plt.axis("off")
    plt.show()

def plot_weight_scatter(y_true, y_pred):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true,y_pred,s=12,alpha=0.6)
    lims = [0, max(max(y_true), max(y_pred))*1.05]
    plt.plot(lims,lims,'r--')
    plt.xlabel("True (g)")
    plt.ylabel("Pred (g)")
    plt.title("Weight Prediction")
    plt.show()
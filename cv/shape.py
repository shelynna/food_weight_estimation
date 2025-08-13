import numpy as np
import cv2
from math import log10
from skimage.measure import moments_hu, regionprops, label
from scipy.stats import entropy

def shape_descriptors(mask: np.ndarray, image: np.ndarray):
    m = (mask>0).astype('uint8')
    if m.sum() == 0:
        return {
            "area_px":0,"perimeter":0,"convex_ratio":0,"eccentricity":0,"aspect_ratio":0,
            "circularity":0,"extent":0,"hu1":0,"hu2":0,"hu3":0,"hu4":0,"hu5":0,"hu6":0,"hu7":0,
            "mean_r":0,"mean_g":0,"mean_b":0,"std_r":0,"std_g":0,"std_b":0,"color_entropy":0
        }
    props = regionprops(label(m))
    # choose largest region
    props.sort(key=lambda x: x.area, reverse=True)
    p = props[0]
    area_px = int(p.area)
    bb_minr, bb_minc, bb_maxr, bb_maxc = p.bbox
    h = bb_maxr - bb_minr
    w = bb_maxc - bb_minc
    aspect_ratio = w / (h+1e-6)
    extent = p.extent
    eccentricity = p.eccentricity
    convex_area = p.convex_area
    convex_ratio = p.area / (convex_area+1e-6)
    # perimeter via contour
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = 0
    for c in cnts: perimeter += cv2.arcLength(c, True)
    circularity = 4 * np.pi * area_px / (perimeter**2 + 1e-6)
    # Hu moments
    moments = cv2.moments(m)
    hu = moments_hu(moments)
    hu_log = [float(-1 * np.sign(h)*log10(abs(h)+1e-12)) for h in hu]
    roi = image[m>0]
    mean_rgb = roi.mean(axis=0)
    std_rgb = roi.std(axis=0)
    hist = cv2.calcHist([roi],[0],None,[32],[0,256])
    hist = hist / (hist.sum()+1e-6)
    color_entropy = float(entropy(hist.flatten()+1e-9, base=2))
    return {
        "area_px": area_px,
        "perimeter": float(perimeter),
        "convex_ratio": float(convex_ratio),
        "eccentricity": float(eccentricity),
        "aspect_ratio": float(aspect_ratio),
        "circularity": float(circularity),
        "extent": float(extent),
        "hu1": hu_log[0],
        "hu2": hu_log[1],
        "hu3": hu_log[2],
        "hu4": hu_log[3],
        "hu5": hu_log[4],
        "hu6": hu_log[5],
        "hu7": hu_log[6],
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "std_r": float(std_rgb[0]),
        "std_g": float(std_rgb[1]),
        "std_b": float(std_rgb[2]),
        "color_entropy": color_entropy
    }

def descriptor_vector(desc: dict):
    keys = ["area_px","perimeter","convex_ratio","eccentricity","aspect_ratio",
            "circularity","extent","hu1","hu2","hu3","hu4","hu5","hu6","hu7",
            "mean_r","mean_g","mean_b","std_r","std_g","std_b","color_entropy"]
    return [desc[k] for k in keys], keys
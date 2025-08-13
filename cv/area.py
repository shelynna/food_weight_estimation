import numpy as np
import cv2

def compute_pixel_area(mask: np.ndarray) -> int:
    """mask binary uint8"""
    return int((mask>0).sum())

def compute_perimeter(mask: np.ndarray) -> float:
    contours,_ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = 0.0
    for c in contours:
        per += cv2.arcLength(c, True)
    return per

def estimate_pixel_to_cm(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Heuristic placeholder: if known plate diameter in cm (e.g., 26 cm) and we detect largest circle.
    Provide override if you have calibration marker.
    """
    # Attempt circle detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,50,param1=100,param2=30,minRadius=50,maxRadius=min(image.shape[:2])//2)
    plate_diameter_cm = 26.0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        r = circles[0,0,2]
        diameter_px = 2*r
        return plate_diameter_cm / diameter_px
    # fallback scale estimation
    mean_dim = np.mean(mask.shape)
    return 26.0 / mean_dim  # naive

def pixel_area_to_cm2(pixel_area:int, pixel_to_cm:float)->float:
    return pixel_area * (pixel_to_cm**2)
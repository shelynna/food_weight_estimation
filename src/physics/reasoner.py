import re

from .priors import DENSITY_PRIORS, THICKNESS_PRIORS, SEMANTIC_MODIFIERS

def extract_semantic_modifiers(mllm_text):
    txt = mllm_text.lower()
    mods=[]
    for k in SEMANTIC_MODIFIERS:
        if re.search(rf"\b{k}\b", txt):
            mods.append(k)
    return mods

def estimate_height(base_thickness, mods):
    factor = 1.0
    for m in mods:
        factor *= SEMANTIC_MODIFIERS[m]
    return base_thickness * factor

def physics_weight(area_cm2, food_class, mllm_text=None):
    density = DENSITY_PRIORS.get(food_class, 1.0)
    base_thickness = THICKNESS_PRIORS.get(food_class, 1.5)
    mods = extract_semantic_modifiers(mllm_text) if mllm_text else []
    height = estimate_height(base_thickness, mods)
    volume_cm3 = area_cm2 * height
    return volume_cm3 * density, {
        "density": density,
        "height_cm": height,
        "volume_cm3": volume_cm3,
        "mods": mods
    }
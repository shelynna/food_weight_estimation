# Priors for density (g/cm^3) and nominal thickness (cm)
# Values are illustrative; adjust with empirical data.
DENSITY_PRIORS = {
    "plantain": 0.95,
    "rice": 0.85,
    "yam": 1.05,
    "fufu": 1.00,
    "stew": 1.10,
    "chicken": 1.08,
    "fish": 1.02
}

THICKNESS_PRIORS = {
    "plantain": 1.8,
    "rice": 2.5,
    "yam": 2.2,
    "fufu": 3.0,
    "stew": 0.8,
    "chicken": 2.5,
    "fish": 1.6
}

SEMANTIC_MODIFIERS = {
    "sliced": 0.7,
    "diced": 0.6,
    "mashed": 0.5,
    "whole": 1.0
}
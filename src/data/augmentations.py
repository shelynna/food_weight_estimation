import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size,img_size, border_mode=0),
        A.RandomBrightnessContrast(0.2,0.2,p=0.5),
        A.HueSaturationValue(10,15,10,p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), translate_percent=(0.0,0.05), rotate=(-10,10), p=0.4),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transforms(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size,img_size, border_mode=0),
        A.Normalize(),
        ToTensorV2()
    ])
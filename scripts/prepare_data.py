import os, json, argparse, glob, shutil
from tqdm import tqdm
from PIL import Image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_split", default=0.8, type=float)
    args=ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    images = glob.glob(os.path.join(args.raw_dir,"*.jpg")) + glob.glob(os.path.join(args.raw_dir,"*.png"))
    images.sort()
    split=int(len(images)*args.train_split)
    train_imgs=images[:split]; val_imgs=images[split:]

    def write_split(name, files):
        split_dir=os.path.join(args.out_dir, name, "images")
        os.makedirs(split_dir, exist_ok=True)
        ann=[]
        for f in tqdm(files, desc=name):
            fname=os.path.basename(f)
            shutil.copy(f, os.path.join(split_dir,fname))
            im=Image.open(f)
            w,h=im.size
            ann.append({"file_name": fname, "width": w, "height": h, "segments_info":[]})
        with open(os.path.join(args.out_dir, name, "annotations.json"),"w") as fw:
            json.dump(ann, fw)
    write_split("train", train_imgs)
    write_split("val", val_imgs)
    # dummy mapping; user should fill real categories.
    class_mapping={"plantain":0,"rice":1,"yam":2,"fufu":3,"stew":4,"chicken":5,"fish":6}
    with open(os.path.join(args.out_dir,"class_mapping.json"),"w") as f:
        json.dump(class_mapping,f)
    print("Prepared dataset skeleton. Populate segments_info and weights later.")

if __name__=="__main__":
    main()
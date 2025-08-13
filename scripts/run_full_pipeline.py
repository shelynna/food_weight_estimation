import argparse, os, subprocess, sys

def run(cmd):
    print("Running:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode!=0:
        raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_all", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.train_all:
        run(f"python training/train_segmentation.py --data_root {args.data_root} --output_dir {args.output_dir}")
        run(f"python training/train_classifier.py --data_root {args.data_root} --output_dir {args.output_dir}")
        run(f"python training/train_weight.py --data_root {args.data_root} --output_dir {args.output_dir}")

if __name__=="__main__":
    main()
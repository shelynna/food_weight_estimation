import yaml, argparse
from dataclasses import dataclass
def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def parse_with_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    for kv in args.override:
        k,v = kv.split("=",1)
        # support nested: a.b.c=val
        d = cfg
        ks = k.split(".")
        for sub in ks[:-1]:
            d = d[sub]
        # type cast basic
        if v.lower() in ["true","false"]:
            v = v.lower()=="true"
        else:
            try:
                if "." in v: v=float(v); 
                else: v=int(v)
            except:
                pass
        d[ks[-1]] = v
    if args.images: cfg.setdefault("inference", {})["images_dir"]=args.images
    return cfg
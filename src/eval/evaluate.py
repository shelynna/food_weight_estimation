import json, argparse, torch
import pandas as pd, numpy as np, os
from ..utils.metrics import regression_metrics, aggregate_instance_to_image
from ..utils.logging import log_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args=ap.parse_args()
    df=pd.read_csv(args.csv)
    rows=df.to_dict("records")
    per_inst=regression_metrics(df.true_weight.values, df.pred_weight.values)
    per_image=aggregate_instance_to_image(rows)
    metrics={**{f"inst_{k}":v for k,v in per_inst.items()},
             **{f"image_{k}":v for k,v in per_image.items()}}
    log_metrics(metrics, args.out)

if __name__=="__main__":
    main()
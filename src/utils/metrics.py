import numpy as np
def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-6)))*100
    return dict(MAE=mae, RMSE=rmse, MAPE=mape)

def classification_accuracy(y_true, y_pred):
    y_true = np.array(y_true); y_pred=np.array(y_pred)
    return float((y_true==y_pred).mean())

def aggregate_instance_to_image(rows):
    # rows: list of dict with 'image_id','pred_weight','true_weight'
    out={}
    for r in rows:
        out.setdefault(r["image_id"], {"pred":0.0,"true":0.0})
        out[r["image_id"]]["pred"] += r["pred_weight"]
        if r.get("true_weight") is not None:
            out[r["image_id"]]["true"] += r["true_weight"]
    preds=[v["pred"] for v in out.values()]
    trues=[v["true"] for v in out.values() if v["true"]>0]
    if len(trues)==len(preds):
        return regression_metrics(trues,preds)
    return {}
from rich.console import Console
from rich.table import Table
import json, os, time
console = Console()

def log_metrics(metrics: dict, path=None):
    table = Table("Metric","Value")
    for k,v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v,(int,float)) else str(v))
    console.print(table)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'w') as f:
            json.dump(metrics,f,indent=2)

class Timer:
    def __init__(self, label): self.label=label
    def __enter__(self):
        self.t=time.time()
    def __exit__(self, *exc):
        console.log(f"{self.label} took {time.time()-self.t:.2f}s")
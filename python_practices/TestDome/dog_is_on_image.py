import numpy as np
import pandas as pd

data = [
    ("TRUE", 91),
    ("TRUE", 23),
    ("TRUE", 76),
    ("FALSE", 48),
    ("FALSE", 36),
    ("FALSE", 36),
    ("TRUE", 92),
    ("TRUE", 88),
    ("TRUE", 51),
    ("FALSE", 10),
    ("FALSE", 28),
    ("FALSE", 62),
    ("TRUE", 75),
    ("TRUE", 80),
    ("TRUE", 54),
    ("FALSE", 72),
    ("FALSE", 22),
    ("FALSE", 50),
    ("FALSE", 12),
    ("TRUE", 29),
    ("TRUE", 59),
    ("TRUE", 78),
    ("FALSE", 32),
    ("TRUE", 93),
]

df = pd.DataFrame(data, columns=["Dog", "Score"])
df["Dog"] = df["Dog"] == "TRUE"

def confusion(threshold):
    pred = df["Score"] >= threshold
    tp = int(((pred) & (df["Dog"])).sum())
    tn = int((~pred & ~df["Dog"]).sum())
    fp = int(((pred) & (~df["Dog"])).sum())
    fn = int((~pred & (df["Dog"])).sum())
    acc = (tp + tn) / len(df)
    return tp, tn, fp, fn, acc

rows = []
for t in [50, 51, 52]:
    tp, tn, fp, fn, acc = confusion(t)
    rows.append({"threshold": t, "TP": tp, "TN": tn, "FP": fp, "FN": fn, "accuracy": acc})

print(pd.DataFrame(rows))


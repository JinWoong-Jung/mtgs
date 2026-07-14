"""Compare graph-only / vlm-only / fixed-blend / learned-router / α-sweep from the raw
logit dump (vlm._dump_logits). All via the SAME build_mtgs_dicts+compute_metrics harness.
CPU only. final_prob = sigmoid( aw*graph_logit + (1-aw)*vlm_logit ), aw = graph weight."""
import math
import torch
from vlm.eval import build_mtgs_dicts, evaluate as ev

C = "/home/jinwoongjung/MTGS/data/vlm_feature"
DUMP = C + "/logits_VLM_Frame_v4_test.pt"
GTM = C + "/gtmeta_test.pt"

d = torch.load(DUMP, weights_only=False)   # {(sid,task,i,j): (graph_logit, vlm_logit, alpha)}
sig = lambda x: 1.0 / (1.0 + math.exp(-x))


def preds_for(mode):
    """mode: 'graph','vlm','learned', or a float graph-weight aw in [0,1]."""
    out = {}
    for k, (g, v, a) in d.items():
        if mode == "graph":
            out[k] = sig(g)
        elif mode == "vlm":
            out[k] = sig(v)
        elif mode == "learned":
            out[k] = sig(a * g + (1 - a) * v)
        else:  # fixed graph-weight
            out[k] = sig(mode * g + (1 - mode) * v)
    return out


def row(name, mode):
    m = ev(build_mtgs_dicts(GTM, preds_for(mode)))
    sap = None
    aps = [m.get(k) for k in ("LAH_AP", "LAEO_AP", "SA_AP")]
    if all(x is not None for x in aps):
        sap = sum(aps) / 3
    print(f"{name:>22} {m['F1_LAH']:>8.4f} {m['F1_LAEO']:>8.4f} {m['AP_SA']:>8.4f} "
          f"{(sap or 0):>10.4f}  | LAH_AP={m['LAH_AP']:.4f} LAEO_AP={m['LAEO_AP']:.4f}")


print(f"[analyze] {len(d)} pairs from {DUMP}")
# mean learned-alpha per task (to see how far the router collapsed)
import collections
al = collections.defaultdict(list)
for (sid, t, i, j), (g, v, a) in d.items():
    al[t].append(a)
print("mean learned α (graph weight): " +
      "  ".join(f"{t}={sum(x)/len(x):.3f}" for t, x in al.items()))

print(f"\n{'config':>22} {'F1_LAH':>8} {'F1_LAEO':>8} {'AP_SA':>8} {'social_ap':>10}")
row("graph-only (aw=1.0)", "graph")
row("vlm-only (aw=0.0)", "vlm")
row("fixed 0.5/0.5", 0.5)
row("fixed aw=0.7", 0.7)
row("fixed aw=0.3", 0.3)
row("learned router", "learned")

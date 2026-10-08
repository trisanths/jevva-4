import json, os, sys
R = "/mnt/nvme/moefy/results/jev"
s = json.load(open(os.path.join(R, "summary.json")))
keys = ["aokvqa", "scienceqa", "ai2d", "mmstar", "mmlu", "arc"]
for n in sys.argv[1:]:
    m = s[n]["metrics"] if n in s else json.load(open(os.path.join(R, n + ".json")))["metrics"]
    print(n.ljust(14), " ".join(f"{k} {100 * m[k]['acc']:.1f}" for k in keys), f"| nll {m['_all']['nll']:.2f} ece {m['_all']['ece']:.3f}")

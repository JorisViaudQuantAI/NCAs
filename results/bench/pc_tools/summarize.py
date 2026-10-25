import json
import sys

for path in sys.argv[1:]:
    rep = json.load(open(path))
    print(f"\n## {path.split('/')[-1]}  batch {rep['train']['batch_size']} chunk {rep['train']['chunk']}")
    print("variant | prec | seg | train ms | full ms | x train | x full | peak GiB | first s | equiv")
    for r in rep["results"]:
        if not r["ok"]:
            print(r["variant"], "bf16" if r["amp"] else "fp32", "FAILED", r.get("error", "")[:120])
            continue
        print(
            f"{r['variant']} | {'bf16' if r['amp'] else 'fp32'} | {r['segment'] or '-'} | "
            f"{r['train']['seconds_mean'] * 1e3:.0f} | {r['full']['seconds_mean'] * 1e3:.0f} | "
            f"{r.get('speedup_vs_reference', float('nan')):.2f} | {r.get('speedup_full_vs_reference', float('nan')):.2f} | "
            f"{r['train'].get('peak_allocated_gb', float('nan')):.2f} | {r['train']['first_iteration_seconds']:.1f} | "
            f"{'ok' if r.get('equivalence_ok', True) else 'FAIL ' + str(r.get('equivalence_error'))[:60]}"
        )

"""Print the pass-mark table (markdown) from results/vision_eval.json and results/audio_eval.json."""
import json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "results"
def ld(n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else None
rows = ["| mark | target | result | controls | verdict |", "|---|---|---|---|---|"]
v = ld("vision_eval.json")
if v:
    pf = lambda b: "**PASS**" if b else "**FAIL**"
    rows.append(f"| V1 retrieval top-1 of 10 (n={v['n']}) | ≥ 60 % | {v['V1_top1']*100:.1f} % | zeroed {v['zeroed_top1']*100:.1f} %, shuffled {v['shuffled_top1']*100:.1f} % (chance 10 %) | {pf(v['V1_top1'] >= .6 and v['zeroed_top1'] < .2 and v['shuffled_top1'] < .2)} |")
    rows.append(f"| V2 held-out caption loss vs no image | ≥ 25 % lower | {v['V2_loss_image']:.3f} vs {v['V2_loss_noimage']:.3f} ({v['V2_reduction']*100:.1f} % lower) | zeroed {v['V2_loss_zeroed']:.3f}, shuffled {v['V2_loss_shuffled']:.3f} | {pf(v['V2_reduction'] >= .25)} |")
    rows.append(f"| V3 50 greedy captions non-degenerate | ≥ 90 % | {v['V3_ok_frac']*100:.0f} % | | {pf(v['V3_ok_frac'] >= .9)} |")
a = ld("audio_eval.json")
if a:
    pf = lambda b: "**PASS**" if b else "**FAIL**"
    rows.append(f"| A1 retrieval top-1 of 10 (n={a['n']}) | ≥ 40 % | {a['A1_top1']*100:.1f} % | zeroed {a['zeroed_top1']*100:.1f} %, shuffled {a['shuffled_top1']*100:.1f} % | {pf(a['A1_top1'] >= .4 and a['zeroed_top1'] < .2 and a['shuffled_top1'] < .2)} |")
    rows.append(f"| A2 held-out caption loss vs no audio | ≥ 15 % lower | {a['A2_loss_audio']:.3f} vs {a['A2_loss_noaudio']:.3f} ({a['A2_reduction']*100:.1f} % lower) | zeroed {a['A2_loss_zeroed']:.3f}, shuffled {a['A2_loss_shuffled']:.3f} | {pf(a['A2_reduction'] >= .15)} |")
    rows.append(f"| A3 30 greedy captions non-degenerate | ≥ 90 % | {a['A3_ok_frac']*100:.0f} % | | {pf(a['A3_ok_frac'] >= .9)} |")
print("\n".join(rows))

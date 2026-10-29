"""Slides for the five-minute video (docs/video/script.md), one per scene, built from the same evidence as the write-up.

Part 2 numbers come from outputs/markets/part2/part2_exhibits.json (scripts/exhibits_part2.py) and its charts;
Part 1 numbers are checked against the write-up text, where each carries its source (`quoted`).
Run: .venv/Scripts/python.exe -I scripts/build_slides.py
Writes docs/video/slides.html: one self-contained page; open it full screen and move with the arrow keys.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "docs" / "writeup" / "writeup.md"
EXH = ROOT / "docs" / "writeup" / "exhibits"
OUT = ROOT / "docs" / "video" / "slides.html"


def quoted(text: str) -> str:
    assert text in WRITEUP.read_text(encoding="utf-8"), f"the write-up does not contain {text!r}"
    return text


def img(name: str) -> str:
    b = base64.b64encode((EXH / name).read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b}' alt=''>"


def rows(r: list[list[str]], head: list[str] | None = None) -> str:
    h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>" if head else ""
    return "<table>" + h + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in r) + "</table>"


def slides() -> list[tuple[str, str, str]]:
    X = json.loads((ROOT / "outputs" / "markets" / "part2" / "part2_exhibits.json").read_text(encoding="utf-8"))["part2_exhibits"]
    C, F = X["common"], X["final"]
    lad, H = C["ladder"], C["holdout"]
    b = next(r for r in F["buildups"] if r["row_id"] == "B00229821")
    k16 = lad["panel2"][2]
    g = F["gender"]
    return [
        ("A technical price for an established book", rows([["Motor", "about 1.0"], ["Buildings", quoted("1.29 to 1.36")], ["Contents", "about 0.84"]], ["Line", "Loss ratio, 2021 to 2024"]),
         "Buildings paid out 1.29 to 1.36 of its premium every year from 2021 to 2024"),
        ("Building the target", "<ul><li>Net cost by loss date; repudiated claims out</li><li>Claims capped: 100k motor, 200k buildings, 30k contents</li>"
                                "<li>Exposure × expected reporting completion: IBNR at row level</li><li>Nine stated data rules; no record excluded</li></ul>",
         "The answer we are scored on is not in the data: it has to be built"),
        ("The keep rule", rows([["Plain GLM", "9.98%"], ["Claims history with credibility", quoted("10.34%")], ["GLM W2", "10.51%"], ["Blend of families", quoted("10.67%")]],
                               ["Step", "Ranking D², 2023"]),
         "A step stays only if it wins in both test years and its interval is above zero"),
        ("The price", f"<p class='big'>500 × 1.058 × 1.009 × 0.9996 = {quoted('533.5')} CU</p><p>cost above the cap × to the 2025 indication × 2025 weather</p>",
         "Expected buildings losses: 1.43 times the current premium"),
        ("The sealed envelope", rows([["Motor", "+24,250"], ["Buildings", "+9,286"], ["Contents", "+7,714"], ["All lines", quoted("+41,250 [2,450, 103,420]")]],
                                     ["Line", "Blend against W2 GLM, 2023"]),
         "Opened once, after the freeze: the blend wins in every line and year"),
        ("Part 1 result", f"<p class='big'>Private {quoted('0.13641')}, 15th</p><p>Winner {quoted('0.13923')}</p>",
         "Asked to explain its recommendation, the AI found a bug in its own code (D16)"),
        ("A new market", rows([["Years of data", "2", "4"], ["New business, training", "14%", ""], ["New business, 2025 book", "76%", ""],
                               ["Gender", "allowed", "not present"]], ["", "Emerging", "Part 1"]),
         "The 2025 book is 76% new business from an acquired broker portfolio"),
        ("Borrow the shape, learn the level", rows([[r["step"], r["v2"] or "", f"{r['d2'][1]:.4f}"] for r in lad["panel1"]], ["Step", "Cross-validation [95%]", "D²"]),
         "Part 1's relativities beat an emerging-only GLM in every line; the emerging data adds only what passes"),
        ("The hold-out decision", img("part2_holdout.png"), "Motor holds; buildings loses in three of four readings and goes back to Part 1's shape (H-F (b))"),
        ("Reviews that found our inconsistency", rows([["Out of time", k16["v1"]], ["Cross-validation", k16["v2"]],
                                                       ["Hold-out, 2024 rows", f"+{H['H1']['k16_motor']['gain']:,.0f}"],
                                                       ["Hold-out, every row", f"+{H['H2']['k16_motor']['gain']:,.0f}"]], ["Motor by peril, motor deviance saved", ""]),
         "Rejected in round 1 on a pooled rule; on the line rule it passes. Final file: 0.15712 public"),
        ("The price of one policy", img("part2_final_buildup.png"),
         f"{b['risk_premium']:.1f} + {b['cat_load']:.1f} + {b['fixed']:.0f} = {b['risk_premium'] + b['cat_load'] + b['fixed']:.1f}, "
         f"÷ {b['divisor']:.2f} = {b['office_premium']:,.1f} CU, against {b['written_premium']:.1f} charged today"),
        ("What the AI got wrong", "<ul><li>D16: a bug in its own completion weights, found when asked why</li><li>D31: wrong about the cause of the gap</li>"
                                  "<li>D36: a wrong diagnosis of a slow run, exposed by a question</li><li>D37: a threshold its own evidence contradicted</li>"
                                  f"<li>D47 to D49: independent reviews found our rule applied inconsistently</li></ul><p>Gender in motor: men {g['men_over_women']:.2f} times women, everything else equal</p>",
         "52 logged exchanges. AI built and tested; the judgments were the actuary's"),
    ]


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video slides</title><style>
:root {{ --ink: #1f2a44; --mute: #5b6475; --bg: #ffffff; --accent: #c0392b; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", Calibri, Arial, sans-serif; }}
section {{ display: none; height: 100vh; box-sizing: border-box; padding: 6vh 8vw; flex-direction: column; }}
section.on {{ display: flex; }}
h1 {{ font-size: 4.2vh; margin: 0 0 3vh; }}
.n {{ color: var(--mute); font-size: 2vh; }}
.body {{ flex: 1; font-size: 3vh; overflow: hidden; }}
.cap {{ border-top: 2px solid var(--accent); padding-top: 1.6vh; font-size: 2.8vh; }}
table {{ border-collapse: collapse; font-size: 2.6vh; }} th, td {{ border-bottom: 1px solid #ccd; padding: 0.8vh 1.6vw; text-align: left; }}
img {{ max-height: 58vh; max-width: 100%; }} .big {{ font-size: 5vh; font-weight: 600; margin: 4vh 0 1vh; }}
</style></head><body>
{body}
<script>
let i = 0; const s = document.querySelectorAll('section');
function show(k) {{ i = Math.max(0, Math.min(s.length - 1, k)); s.forEach((e, j) => e.classList.toggle('on', j === i)); }}
document.addEventListener('keydown', e => {{ if (['ArrowRight', 'PageDown', ' '].includes(e.key)) show(i + 1); if (['ArrowLeft', 'PageUp'].includes(e.key)) show(i - 1); }});
document.addEventListener('click', () => show(i + 1)); show(0);
</script></body></html>"""


def main() -> None:
    S = slides()
    body = "\n".join(f"<section><div class='n'>{k + 1} / {len(S)} · {'Part 1' if k < 6 else 'Part 2'}</div><h1>{t}</h1>"
                     f"<div class='body'>{c}</div><div class='cap'>{cap}</div></section>" for k, (t, c, cap) in enumerate(S))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.format(body=body), encoding="utf-8", newline="\n")
    print("written", OUT, len(S), "slides")


if __name__ == "__main__":
    main()

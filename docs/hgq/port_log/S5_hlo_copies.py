"""Classify the copy ops of a compiled HLO text: computation (ENTRY or other), operand kind, used by the ROOT tuple.

Usage: ``python S5_hlo_copies.py <compiled HLO text>`` (``HLO_DUMP`` of ``S5_g1_probe.py``). Evidence for S5.md "G1".
"""

import collections
import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
comps = re.split(r"\n(?=(?:ENTRY )?%\S+ \(|ENTRY )", text)
res = collections.Counter()
for c in comps:
    head = c.split("\n", 1)[0]
    entry = head.startswith("ENTRY")
    defs = {}
    for line in c.splitlines():
        m = re.match(r"\s*(?:ROOT )?(%\S+) = \S+ (\w[\w-]*)\(", line)
        if m:
            defs[m.group(1)] = m.group(2)
    root = [line for line in c.splitlines() if line.strip().startswith("ROOT")]
    root_ops = set(re.findall(r"%[\w.\-]+", root[0])) if root else set()
    for line in c.splitlines():
        m = re.match(r"\s*(?:ROOT )?(%\S+) = \S+ copy\((%[\w.\-]+)\)", line) or re.match(
            r"\s*(?:ROOT )?(%\S+) = .*? copy\((?:\S+ )?(%[\w.\-]+)\)", line
        )
        if not m:
            continue
        kind = defs.get(m.group(2), "?")
        res["entry" if entry else "nested", kind, "to_root" if m.group(1) in root_ops else "internal"] += 1
print(json.dumps({" / ".join(k): v for k, v in sorted(res.items(), key=lambda kv: -kv[1])}, indent=1))

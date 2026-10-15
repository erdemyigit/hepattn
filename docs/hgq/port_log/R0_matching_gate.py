"""Retarget gate R0(c) / R5: every torch ``tests/matching`` failure is a lap1015 AttributeError.

lap1015 is not built locally, so ``PYTHONPATH=src`` makes ``src/lap1015`` an empty namespace package
(RETARGET_ORG_MAIN.md section 2, gate (c)). Usage::

    pytest -m 'not gpu and not requiresdata' tests/matching --junitxml=matching.xml
    python docs/hgq/port_log/R0_matching_gate.py matching.xml   # exit 0 iff no other failure
"""

import re
import sys
import xml.etree.ElementTree as ET  # noqa: S405  (parses our own pytest junit output)

LAP1015 = re.compile(r"AttributeError: module 'lap1015' has no attribute '(lap_late|lap_early|releases_gil)'")


def main(path: str) -> int:
    failures = []
    for case in ET.parse(path).iter("testcase"):  # noqa: S314  (our own pytest output)
        node = case.find("failure")
        if node is None:
            node = case.find("error")
        if node is not None:
            failures.append((f"{case.get('classname')}::{case.get('name')}", (node.get("message") or "") + (node.text or "")))
    other = [name for name, text in failures if not LAP1015.search(text)]
    print(f"failures={len(failures)} lap1015_attr={len(failures) - len(other)} other={len(other)}")
    for name in other:
        print("OTHER", name)
    return 1 if other else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

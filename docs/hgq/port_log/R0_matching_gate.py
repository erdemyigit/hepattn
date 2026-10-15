"""Retarget gate R0(c) / R5: every torch ``tests/matching`` failure is a lap1015 AttributeError.

lap1015 is not built locally, so ``PYTHONPATH=src`` makes ``src/lap1015`` an empty namespace package
(RETARGET_ORG_MAIN.md section 2, gate (c)). Usage::

    pytest -m 'not gpu and not requiresdata' tests/matching --junitxml=matching.xml
    python docs/hgq/port_log/R0_matching_gate.py matching.xml [--max-failures 106]

Exit 0 iff at least one test ran, every <failure> and <error> message is a lap1015 AttributeError, and
(with ``--max-failures``) the failure count is not above the given bound. Only the ``message`` attribute
is matched: it holds the final exception, so a chained traceback that mentions lap1015 does not count.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET  # noqa: S405  (parses our own pytest junit output)

LAP1015 = re.compile(r"AttributeError: module 'lap1015' has no attribute '(lap_late|lap_early|releases_gil)'")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--max-failures", type=int, default=None)
    args = parser.parse_args(argv)

    ran, failures = 0, []
    for case in ET.parse(args.xml).iter("testcase"):  # noqa: S314  (our own pytest output)
        if case.find("skipped") is None:
            ran += 1
        name = f"{case.get('classname')}::{case.get('name')}"
        failures.extend((name, node.get("message") or "") for tag in ("failure", "error") for node in case.findall(tag))
    other = [name for name, message in failures if not LAP1015.search(message)]
    print(f"ran={ran} failures={len(failures)} lap1015_attr={len(failures) - len(other)} other={len(other)}")
    for name in other:
        print("OTHER", name)
    if ran == 0:
        print("FAIL: no test ran")
        return 1
    if args.max_failures is not None and len(failures) > args.max_failures:
        print(f"FAIL: {len(failures)} failures > --max-failures {args.max_failures}")
        return 1
    return 1 if other else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

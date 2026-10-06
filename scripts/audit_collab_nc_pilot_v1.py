from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_collab_nc_pilot_v1 import phase_audit, phase_manifest, phase_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-audit saved train/validation argmax only; contains no test-evaluation command.")
    parser.add_argument("action", choices=("audit", "report", "manifest"))
    action = parser.parse_args().action
    {"audit": phase_audit, "report": phase_report, "manifest": phase_manifest}[action]()


if __name__ == "__main__":
    main()


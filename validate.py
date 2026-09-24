"""Standalone validation of generated artifacts: python validate.py (exit code 1 on failure)."""
import sys

from rag.checks import run_checks


def main() -> int:
    results = run_checks()
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    failed = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

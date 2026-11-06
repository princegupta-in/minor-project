"""
Check the live status of all Google Earth Engine export tasks.

Usage:
    python scripts/check_gee_tasks.py --project minor-project-510007
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
import ee


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="minor-project-510007")
    p.add_argument("--limit", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    ee.Initialize(project=args.project)

    tasks = ee.data.getTaskList()[:args.limit]
    if not tasks:
        print("No recent tasks found.")
        return

    states = Counter(t["state"] for t in tasks)
    print("\n" + "=" * 70)
    print(f"GEE Task Status Summary (Last {len(tasks)} tasks):")
    for state, count in states.items():
        print(f"  {state:<12}: {count}")
    print("=" * 70)

    print(f"{'Description':<24} | {'State':<10} | {'Task ID':<26}")
    print("-" * 70)
    for t in tasks:
        desc = t.get("description", "Unknown")
        state = t.get("state", "Unknown")
        tid = t.get("id", "")
        print(f"{desc:<24} | {state:<10} | {tid:<26}")
    print("=" * 70)


if __name__ == "__main__":
    main()


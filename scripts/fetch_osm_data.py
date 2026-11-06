import requests
import json
from pathlib import Path
import time

# Focused query: major road corridors and major waterways
query = """
[out:json][timeout:45];
(
  way["highway"~"^(trunk|primary|secondary|tertiary)"](24.155,86.086,24.499,86.863);
  way["waterway"~"^(river|stream)"](24.155,86.086,24.499,86.863);
);
out geom;
"""

endpoints = [
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter"
]

headers = {
    "User-Agent": "DeogharResearchProject/1.0 (academic research)",
    "Accept": "application/json"
}

out_path = Path("data/raw/osm_deoghar.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

success = False
for ep in endpoints:
    print(f"Trying Overpass mirror: {ep}")
    try:
        r = requests.post(ep, data={"data": query}, headers=headers, timeout=50)
        print(f"  Response status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            elements = data.get("elements", [])
            print(f"  Success! Total elements: {len(elements)}")
            highways = [e for e in elements if "highway" in e.get("tags", {})]
            waterways = [e for e in elements if "waterway" in e.get("tags", {})]
            print(f"    Classified Highways: {len(highways)}")
            print(f"    Waterways: {len(waterways)}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
            success = True
            break
        else:
            print(f"  Status {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print(f"  Request failed: {e}")
    time.sleep(1)

if not success:
    print("All mirrors failed or timed out.")

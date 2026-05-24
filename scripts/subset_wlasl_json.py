"""
WLASL JSON Subsetter - Extracts only relevant sign entries from WLASL_v0.3.json.

The full WLASL JSON is ~22MB with 2000+ signs. This extracts just the 8 signs
we need (hello, thank you, thankful, Father, Mother, Yes, No, Help) into a
small ~200KB file for faster processing.

Usage:
    python scripts/subset_wlasl_json.py --input path/to/WLASL_v0.3.json
    python scripts/subset_wlasl_json.py --input path/to/WLASL_v0.3.json --output data/wlasl_subset.json
"""
import json
import argparse
from pathlib import Path

TARGET_SIGNS = [
    'hello', 'thank you', 'thankful',
    'father', 'mother', 'yes', 'no', 'help'
]

SIGN_DISPLAY = {
    'hello': 'hello', 'thank you': 'thanks', 'thankful': 'thanks',
    'father': 'Father', 'mother': 'Mother', 'yes': 'Yes',
    'no': 'No', 'help': 'Help',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to WLASL_v0.3.json')
    parser.add_argument('--output', default='data/wlasl_subset.json', help='Output path for subset JSON')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    size_before = input_path.stat().st_size
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Total entries: {len(data)}")
    print(f"  Size: {size_before / 1024 / 1024:.1f} MB")

    target_lower = set(TARGET_SIGNS)
    subset = [entry for entry in data if entry.get('gloss', '').lower() in target_lower]

    output_data = {
        "metadata": {
            "source": "WLASL_v0.3 subset for Sign-O-Text",
            "target_signs": list(SIGN_DISPLAY.keys()),
            "display_folders": SIGN_DISPLAY,
            "total_entries": len(subset)
        },
        "entries": subset
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    size_after = output_path.stat().st_size
    print(f"\nExtracted {len(subset)} entries to {output_path}")
    print(f"  Size: {size_after / 1024:.1f} KB (from {size_before / 1024 / 1024:.1f} MB)")
    print(f"  Compression ratio: {size_before / size_after:.0f}x")

    print("\nPer-sign breakdown:")
    for entry in subset:
        gloss = entry.get('gloss', '')
        instances = entry.get('instances', [])
        display = SIGN_DISPLAY.get(gloss.lower(), gloss)
        valid_urls = sum(1 for i in instances if i.get('url') and i['url'] != 'N/A')
        print(f"  {display:8s} ({gloss:10s}): {len(instances):3d} instances, {valid_urls} with URLs")

    print(f"\nUsage: python scripts/download_wlasl.py --wlasl-json \"{output_path.resolve()}\"")


if __name__ == '__main__':
    main()

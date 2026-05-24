"""
WLASL Video Downloader for Sign-O-Text
Downloads videos for: hello, thank you, thankful, Father, Mother, Yes, No, Help
"""
import json
import subprocess
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

TARGET_SIGNS = ['hello', 'thank you', 'thankful', 'father', 'mother', 'yes', 'no', 'help']
SIGN_DISPLAY = {
    'hello': 'hello', 'thank you': 'thanks', 'thankful': 'thanks',
    'father': 'Father', 'mother': 'Mother', 'yes': 'Yes',
    'no': 'No', 'help': 'Help',
}


def get_ytdlp_command():
    ytdlp_path = shutil.which('yt-dlp')
    if ytdlp_path:
        return [ytdlp_path]
    if importlib.util.find_spec('yt_dlp'):
        return [sys.executable, '-m', 'yt_dlp']
    raise FileNotFoundError(
        "yt-dlp not found on PATH and module not installed. "
        "Install with: python -m pip install yt-dlp"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wlasl-json', required=True)
    parser.add_argument('--output-dir', default='data/wlasl_videos')
    parser.add_argument('--max-per-sign', type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.wlasl_json, encoding='utf-8') as f:
        raw = json.load(f)

    # Handle both original format (list) and subset format (dict with 'entries' key)
    if isinstance(raw, dict) and 'entries' in raw:
        data = raw['entries']
    elif isinstance(raw, list):
        data = raw
    else:
        print(f"Unknown JSON format: expected list or dict with 'entries' key")
        sys.exit(1)

    ytdlp_cmd = get_ytdlp_command()
    total_ok = total_fail = 0

    for sign_key in TARGET_SIGNS:
        display = SIGN_DISPLAY[sign_key]
        sign_dir = out_dir / display
        sign_dir.mkdir(parents=True, exist_ok=True)

        instances = []
        for entry in data:
            if entry.get('gloss', '').lower() == sign_key:
                instances = entry.get('instances', [])
                break

        if not instances:
            print(f"No instances for {display} (gloss: '{sign_key}')")
            continue

        print(f"--- {display} ({len(instances)} instances) ---")
        downloaded = 0

        for i, inst in enumerate(instances):
            if downloaded >= args.max_per_sign:
                break
            url = inst.get('url')
            vid = inst.get('video_id', f'v{i}')
            if not url or url == 'N/A':
                continue
            out = sign_dir / f'{vid}.mp4'
            if out.exists():
                downloaded += 1
                continue

            try:
                r = subprocess.run(
                    ytdlp_cmd + ['-f', 'best[height<=480]', '-o', str(out),
                                 '--socket-timeout', '30', '--retries', '5',
                                 '--no-warnings', url],
                    capture_output=True, text=True, timeout=120
                )
                if r.returncode == 0:
                    downloaded += 1
                    total_ok += 1
                    sz = out.stat().st_size / 1024 / 1024
                    print(f"  OK {url[:60]}... ({sz:.1f} MB)")
                else:
                    total_fail += 1
                    stderr = r.stderr.strip()[:100] if r.stderr else ''
                    print(f"  FAIL {url[:60]}... {stderr}")
            except Exception as e:
                total_fail += 1
                print(f"  ERROR {url[:60]}... {e}")

        print(f"Got {downloaded} for {display}")

    print(f"Done! OK: {total_ok}, Failed: {total_fail}")
    print("Next: python scripts/process_wlasl_to_npy.py --video-dir data/wlasl_videos")


if __name__ == '__main__':
    main()

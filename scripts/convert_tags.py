import json
import os
import sys

def try_convert(file_path, output_path):
    encodings = ['utf-8', 'big5', 'gbk', 'utf-16', 'utf-8-sig']
    for enc in encodings:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            text = content.decode(enc)
            # Try parsing as JSON to verify
            data = json.loads(text)
            print(f"SUCCESS: Decoded with {enc}. Found {len(data)} tags.")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Failed with {enc}: {str(e)[:50]}")
    return False

if __name__ == "__main__":
    if try_convert("51標籤庫.json", "data/tags.json"):
        print("Conversion successful.")
    else:
        print("All encodings failed.")
        sys.exit(1)

"""Re-save all joblib model files with compression level 3.

Reduces on-disk size by ~60-80% (220MB -> ~50MB) with minimal load-time cost.
Run once: python -m scripts.compress_models
"""
import os
import sys
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'trained')


def compress_all():
    total_before = 0
    total_after = 0

    for city in sorted(os.listdir(MODELS_DIR)):
        city_dir = os.path.join(MODELS_DIR, city)
        if not os.path.isdir(city_dir):
            continue

        city_before = 0
        city_after = 0

        for fname in sorted(os.listdir(city_dir)):
            if not fname.endswith('.joblib'):
                continue

            path = os.path.join(city_dir, fname)
            size_before = os.path.getsize(path)
            city_before += size_before

            # Load and re-save with compression
            obj = joblib.load(path)
            joblib.dump(obj, path, compress=3)

            size_after = os.path.getsize(path)
            city_after += size_after

            ratio = (1 - size_after / size_before) * 100 if size_before > 0 else 0
            print(f"  {fname}: {size_before/1e6:.1f}MB -> {size_after/1e6:.1f}MB ({ratio:.0f}% smaller)")

        total_before += city_before
        total_after += city_after
        ratio = (1 - city_after / city_before) * 100 if city_before > 0 else 0
        print(f"{city}: {city_before/1e6:.1f}MB -> {city_after/1e6:.1f}MB ({ratio:.0f}% smaller)\n")

    ratio = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    print(f"TOTAL: {total_before/1e6:.1f}MB -> {total_after/1e6:.1f}MB ({ratio:.0f}% smaller)")


if __name__ == '__main__':
    compress_all()

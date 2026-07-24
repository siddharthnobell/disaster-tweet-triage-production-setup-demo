"""
Quick latency benchmark for POST /predict.
Run with the FastAPI service already running (uvicorn app.main:app --reload).

    pip install requests   # if not already installed
    python bench_latency.py
"""
import time
import statistics
import requests

URL = "http://127.0.0.1:8000/predict"

SAMPLES = [
    {"text": "7.2 magnitude earthquake hits coastal region, thousands evacuated"},
    {"text": "Wildfire forces evacuation of entire mountain town overnight"},
    {"text": "just beat my personal best at the gym today feeling great"},
    {"text": "Flash floods sweep through downtown streets after record rainfall"},
    {"text": "this traffic jam is a total disaster, going to be late again"},
    {"text": "can't believe how good this new pizza place is"},
    {"text": "Magnitude 7 earthquake strikes, buildings collapse, rescue teams deployed"},
    {"text": "this exam was an absolute disaster lol"},
    {"text": "Tornado warning issued for the entire county until further notice"},
    {"text": "my boss's decision was an absolute disaster for team morale"},
]

N_ROUNDS = 5  # 5 x 10 = 50 requests total

latencies_ms = []
errors = 0

for _ in range(N_ROUNDS):
    for payload in SAMPLES:
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, json=payload, timeout=5)
            r.raise_for_status()
        except Exception:
            errors += 1
            continue
        latencies_ms.append((time.perf_counter() - t0) * 1000)

latencies_ms.sort()
n = len(latencies_ms)
avg = statistics.mean(latencies_ms)
p50 = latencies_ms[int(0.50 * n)]
p95 = latencies_ms[int(0.95 * n)]
p99 = latencies_ms[min(int(0.99 * n), n - 1)]

print(f"requests sent : {n + errors}")
print(f"successful    : {n}")
print(f"errors        : {errors}")
print(f"avg latency   : {avg:.1f} ms")
print(f"p50 latency   : {p50:.1f} ms")
print(f"p95 latency   : {p95:.1f} ms")
print(f"p99 latency   : {p99:.1f} ms")
print(f"min / max     : {latencies_ms[0]:.1f} ms / {latencies_ms[-1]:.1f} ms")
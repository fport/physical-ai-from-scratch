"""Appendix C · Inference latency of small PyTorch policies on a Mac: CPU vs MPS (Apple GPU).

Two policies of the size Part 7 trains:
  mlp:  state (64) -> 256 -> 256 -> action (6)                      (~85k parameters)
  cnn:  image 3x96x96 -> 4 conv layers -> 256 -> action chunk 50x6   (vision policy head)
Sweep: batch size -> median latency of one forward pass (no grad), CPU vs MPS.
MPS timings include torch.mps.synchronize(), i.e. the time until the result is usable.

The machine was heavily loaded while this ran: numbers show the shape of the trade-off, not a benchmark.
Run:  python experiment.py
"""
import platform
import statistics
import time

import torch
from torch import nn

torch.manual_seed(0)


def mlp():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6))


def cnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 50 * 6))


def time_forward(model, x, device, n=50, warmup=10):
    model, x = model.to(device).eval(), x.to(device)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    ts = []
    with torch.no_grad():
        for i in range(warmup + n):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            if i >= warmup:
                ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000, min(ts) * 1000


def main():
    mps = torch.backends.mps.is_available()
    print(f"torch {torch.__version__} · python {platform.python_version()} · {platform.machine()} · "
          f"{platform.mac_ver()[0] or platform.system()} · MPS available: {mps} · CPU threads {torch.get_num_threads()}")
    print("median (min) latency of one forward pass in ms, 50 timed runs after 10 warm-up runs; heavily loaded "
          "machine, not a benchmark")
    for name, make, shape in (("mlp", mlp, (64,)), ("cnn", cnn, (3, 96, 96))):
        model = make()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n{name} ({n_par} parameters), input {list(shape)}")
        print(f"{'batch':>7}{'CPU ms':>22}{'MPS ms':>20}{'CPU / MPS':>11}")
        for b in (1, 8, 64, 256):
            x = torch.randn(b, *shape)
            c_med, c_min = time_forward(model, x, "cpu")
            if mps:
                g_med, g_min = time_forward(model, x, "mps")
                model = model.to("cpu")
                print(f"{b:>7}{f'{c_med:.3f} ({c_min:.3f})':>22}{f'{g_med:.3f} ({g_min:.3f})':>20}{c_med / g_med:>11.2f}")
            else:
                print(f"{b:>7}{f'{c_med:.3f} ({c_min:.3f})':>22}{'-':>20}{'-':>11}")


if __name__ == "__main__":
    main()

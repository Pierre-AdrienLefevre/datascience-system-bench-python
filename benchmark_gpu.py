import time

import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def gpu_synchronize():
    """Synchronise le GPU pour s'assurer que toutes les opérations sont terminées."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def gpu_benchmark_pytorch(timeout=120, size=3_000):
    """
    Benchmark GPU : multiplication de matrices + sin sur des matrices size x size.
    Tourne pendant 'timeout' secondes et compte les opérations complétées.
    Sync GPU à chaque op pour un time check précis (~0.05-0.1s par op).
    Score = ops/s (opérations par seconde).
    """
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = "cuda" if torch.cuda.is_available() else "mps"

        matrix_a = torch.rand((size, size), device=device)
        matrix_b = torch.rand((size, size), device=device)

        gpu_synchronize()

        deadline = time.time() + timeout
        completed = 0
        start = time.time()

        with Progress(
                TextColumn(f"[bold cyan]GPU ({device.upper()}, {size}x{size})"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn(
                    "[bold]{task.fields[ops]}[/bold] ops | [bold green]{task.fields[ops_s]:.2f}[/bold green] ops/s"),
        ) as progress:
            task = progress.add_task("gpu", total=timeout, ops=0, ops_s=0.0)
            while time.time() < deadline:
                result = torch.matmul(matrix_a, matrix_b)
                result = torch.sin(result)
                gpu_synchronize()
                completed += 1
                elapsed = time.time() - start
                progress.update(task, completed=min(elapsed, timeout), ops=completed, ops_s=completed / elapsed)

        duration = time.time() - start
        ops_per_sec = completed / duration if duration > 0 else 0

        return {
            "duration_seconds": round(duration, 2),
            "timeout_seconds": timeout,
            "iterations_completed": completed,
            "ops_per_sec": round(ops_per_sec, 4),
            "matrix_size": size,
        }
    else:
        print("No GPU available for PyTorch.")
        return None

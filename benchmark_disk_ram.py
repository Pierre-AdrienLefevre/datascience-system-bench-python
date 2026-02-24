import os
import tempfile
import time

import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def ram_benchmark(timeout=120, size=50_000_000):
    """
    Benchmark RAM : alloue et somme un array NumPy de 'size' elements en boucle.
    Tourne pendant 'timeout' secondes et compte les iterations completees.
    Chaque op ~380MB (50M float64), prend <0.1s pour un depassement negligeable.
    Score = ops/s (iterations par seconde).
    """
    bytes_per_element = 8  # float64
    mb_per_iter = (size * bytes_per_element) / (1024 ** 2)
    gb_per_iter = mb_per_iter / 1024

    deadline = time.time() + timeout
    completed = 0
    start = time.time()

    with Progress(
            TextColumn(f"[bold cyan]RAM ({mb_per_iter:.0f}MB/op)"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[bold]{task.fields[ops]}[/bold] ops | [bold green]{task.fields[gb_s]:.2f}[/bold green] GB/s"),
    ) as progress:
        task = progress.add_task("ram", total=timeout, ops=0, gb_s=0.0)
        while time.time() < deadline:
            data = np.random.rand(size)
            np.sum(data)
            del data
            completed += 1
            elapsed = time.time() - start
            total_gb = completed * gb_per_iter
            progress.update(task, completed=min(elapsed, timeout), ops=completed, gb_s=total_gb / elapsed)

    duration = time.time() - start
    ops_per_sec = completed / duration if duration > 0 else 0
    total_gb = completed * gb_per_iter
    gb_per_sec = total_gb / duration if duration > 0 else 0

    return {
        "duration_seconds": round(duration, 2),
        "timeout_seconds": timeout,
        "iterations_completed": completed,
        "ops_per_sec": round(ops_per_sec, 4),
        "total_gb_processed": round(total_gb, 2),
        "throughput_gb_per_sec": round(gb_per_sec, 2),
    }


def disk_benchmark(timeout=120, file_size=50_000_000):
    """
    Benchmark Disk : ecrit et lit un fichier de 'file_size' octets en boucle.
    Les donnees sont pre-generees pour ne mesurer que l'I/O pur.
    Chaque cycle ~50MB write+read, prend <0.1s pour un depassement negligeable.
    Score = ops/s (cycles write+read par seconde).
    """
    mb_per_iter = file_size / (1024 ** 2)
    gb_per_iter = file_size / (1024 ** 3)

    # Pre-generer les donnees pour ne mesurer que l'I/O
    random_data = os.urandom(file_size)

    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix="benchmark_")
    temp_path = temp_file.name
    temp_file.close()

    try:
        deadline = time.time() + timeout
        completed = 0
        total_write_time = 0
        total_read_time = 0
        start = time.time()

        with Progress(
                TextColumn(f"[bold cyan]Disk ({mb_per_iter:.0f}MB/op)"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TextColumn(
                    "[bold]{task.fields[ops]}[/bold] cycles | W:[bold green]{task.fields[w_gbs]:.2f}[/bold green] R:[bold green]{task.fields[r_gbs]:.2f}[/bold green] GB/s"),
        ) as progress:
            task = progress.add_task("disk", total=timeout, ops=0, w_gbs=0.0, r_gbs=0.0)
            while time.time() < deadline:
                # Write
                w_start = time.time()
                with open(temp_path, "wb") as f:
                    f.write(random_data)
                w_end = time.time()

                # Read
                with open(temp_path, "rb") as f:
                    _ = f.read()
                r_end = time.time()

                total_write_time += w_end - w_start
                total_read_time += r_end - w_end
                completed += 1

                elapsed = time.time() - start
                total_gb = completed * gb_per_iter
                w_gbs = total_gb / total_write_time if total_write_time > 0 else 0
                r_gbs = total_gb / total_read_time if total_read_time > 0 else 0
                progress.update(task, completed=min(elapsed, timeout), ops=completed, w_gbs=w_gbs, r_gbs=r_gbs)

        duration = time.time() - start
        ops_per_sec = completed / duration if duration > 0 else 0
        total_gb = completed * gb_per_iter
        write_gb_per_sec = total_gb / total_write_time if total_write_time > 0 else 0
        read_gb_per_sec = total_gb / total_read_time if total_read_time > 0 else 0

        return {
            "duration_seconds": round(duration, 2),
            "timeout_seconds": timeout,
            "iterations_completed": completed,
            "ops_per_sec": round(ops_per_sec, 4),
            "total_gb_processed": round(total_gb, 2),
            "write_throughput_gb_per_sec": round(write_gb_per_sec, 2),
            "read_throughput_gb_per_sec": round(read_gb_per_sec, 2),
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

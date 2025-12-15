import os
import tempfile
import time

import numpy as np


def ram_benchmark(size=2_000_000_000):  # 2 milliards d'éléments (~15 GB)
    print("Starting RAM benchmark...")
    start = time.time()
    data = np.random.rand(size)
    sum_data = np.sum(data)
    end = time.time()
    print(f"RAM benchmark completed in {end - start:.2f} seconds (Sum: {sum_data})")
    return end - start


def disk_benchmark(file_size=4_000_000_000):  # Fichier de 4GB
    print("Starting Disk benchmark...")

    # Utiliser un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix="benchmark_")
    temp_path = temp_file.name
    temp_file.close()

    try:
        start = time.time()
        with open(temp_path, "wb") as f:
            f.write(os.urandom(file_size))
        write_time = time.time()
        with open(temp_path, "rb") as f:
            _ = f.read()
        read_time = time.time()

        write_duration = write_time - start
        read_duration = read_time - write_time
        total_duration = read_time - start

        print(f"Disk Write: {write_duration:.2f}s, Read: {read_duration:.2f}s, Total: {total_duration:.2f}s")
        return total_duration
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

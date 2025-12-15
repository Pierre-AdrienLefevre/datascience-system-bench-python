import argparse
import csv
import json
import os
import platform
import socket
import time
from datetime import datetime
from threading import Thread

import psutil
import torch

from benchmark_cpu import cpu_benchmark_singlecore, cpu_benchmark_multicore
from benchmark_disk_ram import ram_benchmark, disk_benchmark
from benchmark_gpu import gpu_benchmark_pytorch

# Timestamp unique pour la session
SESSION_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_DIR = "results"
SESSION_FILENAME = None  # Sera défini après get_system_info()


def ensure_results_dir():
    """Crée le dossier results/ s'il n'existe pas."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def sanitize_filename(name):
    """Nettoie un nom pour l'utiliser dans un nom de fichier."""
    # Remplacer les caractères problématiques
    for char in [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']:
        name = name.replace(char, '-')
    return name


def get_short_gpu_name():
    """Retourne un nom court pour le GPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        # Simplifier les noms courants
        name = name.replace("NVIDIA ", "").replace("GeForce ", "")
        name = name.replace(" ", "")
        return name
    elif torch.backends.mps.is_available():
        return "AppleMetal"
    return "NoGPU"


def generate_session_filename(system_info):
    """Génère un nom de fichier descriptif basé sur les infos système."""
    hostname = sanitize_filename(system_info["hostname"])
    cpu_cores = system_info["cpu"]["physical_cores"]
    ram_gb = int(system_info["memory"]["total"])
    gpu = sanitize_filename(get_short_gpu_name())

    return f"{hostname}_{cpu_cores}cores_{ram_gb}GB_{gpu}_{SESSION_TIMESTAMP}"


def get_system_info():
    """
    Collecte les informations détaillées du système.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        },
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
            "available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
    }

    # Informations GPU détaillées
    if torch.cuda.is_available():
        info["gpu"] = {
            "type": "CUDA",
            "name": torch.cuda.get_device_name(),
            "memory_total": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),  # GB
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        }
    elif torch.backends.mps.is_available():
        info["gpu"] = {
            "type": "MPS",
            "name": "Apple Metal",
        }
    else:
        info["gpu"] = {
            "type": "None",
            "name": "No GPU acceleration available"
        }

    return info


def get_log_file_path(extension="log"):
    """Retourne le chemin du fichier de log pour cette session."""
    global SESSION_FILENAME
    if SESSION_FILENAME:
        return os.path.join(RESULTS_DIR, f"{SESSION_FILENAME}.{extension}")
    # Fallback si SESSION_FILENAME n'est pas encore défini
    return os.path.join(RESULTS_DIR, f"benchmark_results-{SESSION_TIMESTAMP}.{extension}")


def log_benchmark_result(test_name, duration, system_info, log_file=None):
    """
    Enregistre les résultats de benchmark dans un fichier log.
    """
    if log_file is None:
        log_file = get_log_file_path("log")

    log_entry = {
        "test_name": test_name,
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info
    }

    # Ajouter au fichier log (format JSON Lines)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"✅ {test_name}: {duration:.2f}s - Logged to {log_file}")


def export_to_csv(results, system_info):
    """
    Exporte les résultats en format CSV.
    """
    csv_file = get_log_file_path("csv")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # En-tête avec infos système
        writer.writerow(["# Benchmark Results"])
        writer.writerow(["# Hostname", system_info["hostname"]])
        writer.writerow(["# Platform", f"{system_info['platform']['system']} {system_info['platform']['machine']}"])
        writer.writerow(["# CPU Cores",
                         f"{system_info['cpu']['physical_cores']} physical, {system_info['cpu']['logical_cores']} logical"])
        writer.writerow(["# RAM Total", f"{system_info['memory']['total']} GB"])
        writer.writerow(["# GPU", system_info["gpu"]["name"]])
        writer.writerow(["# Timestamp", SESSION_TIMESTAMP])
        writer.writerow([])

        # Données
        writer.writerow(["Test Name", "Duration (seconds)"])
        for test_name, duration in results.items():
            writer.writerow([test_name, round(duration, 2)])

    print(f"📊 CSV exported to {csv_file}")

def combined_cpu_gpu_benchmark(cpu_iterations=8_000_000_000, gpu_size=15_000, gpu_loops=400, n_jobs=None, loops=50):
    """
    Benchmark combiné CPU + GPU.
    - cpu_iterations : Nombre total d'itérations pour le CPU.
    - gpu_size : Taille des matrices pour le GPU.
    - gpu_loops : Nombre de répétitions pour le GPU.
    - n_jobs : Nombre de cœurs utilisés pour le CPU (None = tous les cœurs).
    - loops : Nombre de boucles pour prolonger le test.
    """
    print("Starting combined CPU + GPU benchmark...")

    # CPU benchmark (multicore)
    def cpu_benchmark():
        print("Running CPU tasks...")
        cpu_benchmark_multicore(cpu_iterations, n_jobs=n_jobs, loops=loops)

    # GPU benchmark
    def gpu_benchmark():
        print("Running GPU tasks...")
        gpu_benchmark_pytorch(gpu_size, gpu_loops)

    # Start CPU and GPU benchmarks in parallel
    cpu_thread = Thread(target=cpu_benchmark)
    gpu_thread = Thread(target=gpu_benchmark)

    start = time.time()
    cpu_thread.start()
    gpu_thread.start()

    # Wait for both to finish
    cpu_thread.join()
    gpu_thread.join()
    end = time.time()

    duration = end - start
    print(f"Combined CPU + GPU benchmark completed in {duration:.2f} seconds")
    return duration


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Benchmark tool for evaluating machine performance for data science tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all benchmarks
  python main.py --only cpu         # Run only CPU benchmarks
  python main.py --only gpu disk    # Run only GPU and Disk benchmarks
  python main.py --skip combined    # Skip the combined CPU+GPU benchmark
  python main.py --no-csv           # Disable CSV export
        """
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=["cpu-single", "cpu-multi", "ram", "disk", "gpu", "combined"],
        help="Run only specified benchmarks"
    )

    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["cpu-single", "cpu-multi", "ram", "disk", "gpu", "combined"],
        default=[],
        help="Skip specified benchmarks"
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV export"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks and exit"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Mapping des noms CLI vers les benchmarks
    benchmark_map = {
        "cpu-single": ("CPU Single Core", cpu_benchmark_singlecore),
        "cpu-multi": ("CPU Multi Core", cpu_benchmark_multicore),
        "ram": ("RAM", ram_benchmark),
        "disk": ("Disk", disk_benchmark),
        "gpu": ("GPU", gpu_benchmark_pytorch),
        "combined": ("Combined CPU + GPU", combined_cpu_gpu_benchmark),
    }

    if args.list:
        print("Available benchmarks:")
        for key, (name, _) in benchmark_map.items():
            print(f"  {key:12} -> {name}")
        return

    # Créer le dossier results/
    ensure_results_dir()

    # Display system information
    print("=== System Information ===")
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Metal) available")
    else:
        print("No GPU acceleration available")
    print("=" * 30)

    # Collecte des informations système et génération du nom de fichier
    system_info = get_system_info()

    global SESSION_FILENAME
    SESSION_FILENAME = generate_session_filename(system_info)
    print(f"📁 Output files: {SESSION_FILENAME}.*")

    # Sélection des benchmarks à exécuter
    if args.only:
        benchmarks = [(name, func) for key, (name, func) in benchmark_map.items() if key in args.only]
    else:
        benchmarks = [(name, func) for key, (name, func) in benchmark_map.items() if key not in args.skip]

    print(f"\n🚀 Starting {len(benchmarks)} benchmark(s)...")
    results = {}

    for test_name, test_func in benchmarks:
        print(f"\n--- Running {test_name} ---")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time

            if result is not None:
                results[test_name] = duration
                log_benchmark_result(test_name, duration, system_info)
            else:
                print(f"❌ {test_name}: Failed (no GPU available)")

        except Exception as e:
            print(f"❌ {test_name}: Error - {str(e)}")

    # Export CSV si activé
    if not args.no_csv and results:
        export_to_csv(results, system_info)

    print("\n📊 Final Results Summary:")
    for test_name, duration in results.items():
        print(f"  {test_name}: {duration:.2f}s")

    print(f"\n📝 Detailed results logged to: {get_log_file_path('log')}")
    if not args.no_csv and results:
        print(f"📊 CSV results exported to: {get_log_file_path('csv')}")
    print(f"🖥️  Machine: {system_info['hostname']} ({system_info['platform']['system']} {system_info['platform']['machine']})")
    print(f"⏰ Session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

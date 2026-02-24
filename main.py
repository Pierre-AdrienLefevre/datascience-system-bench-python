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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    for char in [' ', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '.']:
        name = name.replace(char, '-')
    return name


def get_short_gpu_name():
    """Retourne un nom court pour le GPU."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name()
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
            "memory_total": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
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
    return os.path.join(RESULTS_DIR, f"benchmark_results-{SESSION_TIMESTAMP}.{extension}")


def log_benchmark_result(test_name, result_data, system_info, log_file=None):
    """
    Enregistre les résultats de benchmark dans un fichier log (JSON Lines).
    result_data est un dict contenant duration_seconds, ops_per_sec, etc.
    """
    if log_file is None:
        log_file = get_log_file_path("log")

    log_entry = {
        "test_name": test_name,
        **result_data,
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def export_to_csv(results, system_info):
    """
    Exporte les résultats en format CSV.
    results est un dict {test_name: result_data_dict}.
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
        writer.writerow(["Test Name", "Duration (s)", "Timeout (s)", "Iterations", "Ops/s"])
        for test_name, result_data in results.items():
            writer.writerow([
                test_name,
                round(result_data["duration_seconds"], 2),
                result_data.get("timeout_seconds", "N/A"),
                result_data.get("iterations_completed", "N/A"),
                result_data.get("ops_per_sec", "N/A"),
            ])

    print(f"  CSV exported to {csv_file}")


def combined_cpu_gpu_benchmark(timeout=120):
    """
    Benchmark combiné CPU + GPU en parallèle.
    Les deux benchmarks tournent simultanément pendant 'timeout' secondes.
    """
    cpu_result = {}
    gpu_result = {}

    def cpu_work():
        nonlocal cpu_result
        cpu_result = cpu_benchmark_multicore(timeout=timeout)

    def gpu_work():
        nonlocal gpu_result
        gpu_result = gpu_benchmark_pytorch(timeout=timeout)

    cpu_thread = Thread(target=cpu_work)
    gpu_thread = Thread(target=gpu_work)

    start = time.time()
    cpu_thread.start()
    gpu_thread.start()

    cpu_thread.join()
    gpu_thread.join()
    duration = time.time() - start

    return {
        "duration_seconds": round(duration, 2),
        "timeout_seconds": timeout,
        "iterations_completed": 1,
        "ops_per_sec": round(1 / duration, 4) if duration > 0 else 0,
        "cpu_iterations": cpu_result.get("iterations_completed", 0),
        "cpu_ops_per_sec": cpu_result.get("ops_per_sec", 0),
        "gpu_iterations": gpu_result.get("iterations_completed", 0) if gpu_result else 0,
        "gpu_ops_per_sec": gpu_result.get("ops_per_sec", 0) if gpu_result else 0,
    }


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Benchmark tool for evaluating machine performance for data science tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run all benchmarks (120s each)
  python main.py --cpu-single-timeout 60      # CPU single-core with 60s timeout
  python main.py --gpu-timeout 300            # GPU with 5 min timeout
  python main.py --only cpu-single gpu        # Run only CPU single-core and GPU
  python main.py --skip combined              # Skip the combined CPU+GPU benchmark
  python main.py --no-csv                     # Disable CSV export
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

    # Timeouts par benchmark
    parser.add_argument(
        "--cpu-single-timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Timeout for CPU single-core benchmark (default: 30)"
    )
    parser.add_argument(
        "--cpu-multi-timeout",
        type=int,
        default=240,
        metavar="SECONDS",
        help="Timeout for CPU multi-core benchmark (default: 240)"
    )
    parser.add_argument(
        "--gpu-timeout",
        type=int,
        default=240,
        metavar="SECONDS",
        help="Timeout for GPU benchmark (default: 240)"
    )
    parser.add_argument(
        "--ram-timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Timeout for RAM benchmark (default: 30)"
    )
    parser.add_argument(
        "--disk-timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Timeout for Disk benchmark (default: 30)"
    )
    parser.add_argument(
        "--combined-timeout",
        type=int,
        default=240,
        metavar="SECONDS",
        help="Timeout for combined CPU+GPU benchmark (default: 240)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()

    # Mapping des noms CLI vers les benchmarks avec leurs timeouts
    benchmark_map = {
        "cpu-single": ("CPU Single Core", lambda: cpu_benchmark_singlecore(timeout=args.cpu_single_timeout)),
        "cpu-multi": ("CPU Multi Core", lambda: cpu_benchmark_multicore(timeout=args.cpu_multi_timeout)),
        "ram": ("RAM", lambda: ram_benchmark(timeout=args.ram_timeout)),
        "disk": ("Disk", lambda: disk_benchmark(timeout=args.disk_timeout)),
        "gpu": ("GPU", lambda: gpu_benchmark_pytorch(timeout=args.gpu_timeout)),
        "combined": ("Combined CPU + GPU", lambda: combined_cpu_gpu_benchmark(timeout=args.combined_timeout)),
    }

    if args.list:
        print("Available benchmarks:")
        for key, (name, _) in benchmark_map.items():
            print(f"  {key:12} -> {name}")
        return

    # Créer le dossier results/
    ensure_results_dir()

    # Collecte des informations système et génération du nom de fichier
    system_info = get_system_info()

    global SESSION_FILENAME
    SESSION_FILENAME = generate_session_filename(system_info)

    # Afficher les infos système avec un Panel rich
    gpu_info = system_info["gpu"]["name"]
    if torch.cuda.is_available():
        gpu_info += f" ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"

    sys_text = (
        f"[bold]Hostname:[/bold] {system_info['hostname']}\n"
        f"[bold]Platform:[/bold] {system_info['platform']['system']} {system_info['platform']['machine']}\n"
        f"[bold]CPU:[/bold] {system_info['cpu']['physical_cores']} physical / {system_info['cpu']['logical_cores']} logical cores\n"
        f"[bold]RAM:[/bold] {system_info['memory']['total']} GB\n"
        f"[bold]GPU:[/bold] {gpu_info}\n"
        f"[bold]PyTorch:[/bold] {torch.__version__}"
    )
    console.print(Panel(sys_text, title="[bold cyan]System Information", border_style="cyan"))

    # Sélection des benchmarks à exécuter
    if args.only:
        benchmarks = [(key, name, func) for key, (name, func) in benchmark_map.items() if key in args.only]
    else:
        benchmarks = [(key, name, func) for key, (name, func) in benchmark_map.items() if key not in args.skip]

    timeout_map = {
        "cpu-single": args.cpu_single_timeout,
        "cpu-multi": args.cpu_multi_timeout,
        "ram": args.ram_timeout,
        "disk": args.disk_timeout,
        "gpu": args.gpu_timeout,
        "combined": args.combined_timeout,
    }

    # Afficher le plan d'exécution
    plan_table = Table(title=f"Running {len(benchmarks)} benchmark(s)", show_header=True, header_style="bold")
    plan_table.add_column("Benchmark", style="cyan")
    plan_table.add_column("Timeout", justify="right")
    for key, name, _ in benchmarks:
        plan_table.add_row(name, f"{timeout_map[key]}s")
    console.print(plan_table)
    console.print()

    results = {}

    for key, test_name, test_func in benchmarks:
        try:
            result_data = test_func()

            if result_data is not None:
                results[test_name] = result_data
                log_benchmark_result(test_name, result_data, system_info)
            else:
                console.print(f"  [red]{test_name}: Failed (no GPU available)[/red]")

        except Exception as e:
            console.print(f"  [red]{test_name}: Error - {str(e)}[/red]")

        console.print()

    # Export CSV si activé
    if not args.no_csv and results:
        export_to_csv(results, system_info)

    # Tableau récapitulatif avec rich
    summary = Table(title="Final Results", show_header=True, header_style="bold green")
    summary.add_column("Test", style="cyan")
    summary.add_column("Duration", justify="right")
    summary.add_column("Timeout", justify="right")
    summary.add_column("Iterations", justify="right")
    summary.add_column("Ops/s", justify="right", style="bold green")
    for test_name, result_data in results.items():
        summary.add_row(
            test_name,
            f"{result_data['duration_seconds']:.2f}s",
            f"{result_data.get('timeout_seconds', 'N/A')}s",
            str(result_data.get("iterations_completed", "N/A")),
            str(result_data.get("ops_per_sec", "N/A")),
        )
    console.print(summary)

    console.print(f"\n[dim]Log:[/dim] {get_log_file_path('log')}")
    if not args.no_csv and results:
        console.print(f"[dim]CSV:[/dim] {get_log_file_path('csv')}")
    console.print(
        f"[dim]Machine:[/dim] {system_info['hostname']} ({system_info['platform']['system']} {system_info['platform']['machine']})")
    console.print(f"[dim]Completed:[/dim] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

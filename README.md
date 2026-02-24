# System Performance Benchmark Tool | Outil de Benchmark de Performance Systeme

[English](#english) | [Francais](#francais)

---

## English

### Overview

A comprehensive system performance benchmarking tool designed specifically for **Data Science workloads**. This tool
measures performance across multiple dimensions: CPU (single/multi-core), RAM, Disk I/O, GPU, and combined CPU+GPU
workloads.

Each benchmark runs for a **configurable duration** and reports a **throughput score (ops/s)** for easy cross-machine
comparison. Progress is displayed with modern rich progress bars.

### Features

- **CPU Benchmarks**: Data Science-oriented tasks including:
    - Linear regression training with least squares
    - K-Means clustering algorithm (8 clusters, 20 iterations)
    - Feature engineering & preprocessing (PCA, polynomial features)
    - Gradient descent optimization (multi-start, 20D)

- **GPU Benchmarks**: PyTorch-based matrix operations
    - 3,000x3,000 matrix multiplications + sin
    - GPU sync every op for precise timing
    - Automatic CUDA/MPS backend detection

- **Memory & Storage**: Large-scale NumPy operations and file I/O
    - RAM: 380MB array allocation + summation per op
    - Disk: 50MB write+read per op (pre-generated data, pure I/O)

- **Modern UI**: Rich progress bars with live ops/s counter, panels, and tables

- **Flexible Timing**: Per-benchmark timeout configuration (ops are <0.1s so timeout is respected precisely)

- **Detailed Logging**: JSON Lines + CSV output with full system information

### Requirements

- **Python**: 3.14+
- **Package Manager**: uv (recommended)
- **OS**: Windows, macOS, Linux
- **Hardware**: Multi-core CPU, Optional GPU (NVIDIA CUDA or Apple Metal)

### Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd Benchmark
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run benchmarks**:
   ```bash
   uv run python main.py
   ```

### CLI Options

```
--only BENCHMARK [...]       Run only specified benchmarks
--skip BENCHMARK [...]       Skip specified benchmarks
--no-csv                     Disable CSV export
--list                       List available benchmarks and exit

--cpu-single-timeout SECONDS CPU single-core timeout (default: 30)
--cpu-multi-timeout SECONDS  CPU multi-core timeout (default: 240)
--gpu-timeout SECONDS        GPU timeout (default: 240)
--ram-timeout SECONDS        RAM timeout (default: 30)
--disk-timeout SECONDS       Disk timeout (default: 30)
--combined-timeout SECONDS   Combined CPU+GPU timeout (default: 240)
```

Available benchmarks: `cpu-single`, `cpu-multi`, `ram`, `disk`, `gpu`, `combined`

### Examples

```bash
# Run all benchmarks with default timeouts
uv run python main.py

# Quick test with short timeouts
uv run python main.py --cpu-single-timeout 10 --gpu-timeout 30

# Run only GPU benchmark with 5 minutes
uv run python main.py --only gpu --gpu-timeout 300

# Skip combined benchmark
uv run python main.py --skip combined
```

### Default Durations

| Benchmark        | Default Timeout |
|------------------|:---------------:|
| CPU Single Core  |       30s       |
| CPU Multi Core   |      240s       |
| GPU              |      240s       |
| RAM              |       30s       |
| Disk             |       30s       |
| Combined CPU+GPU |      240s       |

### Output

The tool generates:

- **Console output**: Rich progress bars with live ops/s, summary table
- **Log files**: `results/<hostname>_<cores>cores_<ram>GB_<gpu>_<timestamp>.log` (JSON Lines)
- **CSV files**: `results/<hostname>_<cores>cores_<ram>GB_<gpu>_<timestamp>.csv`

### Dependencies

- **NumPy**: Mathematical operations and arrays
- **PyTorch**: GPU computations with CUDA/MPS support
- **joblib**: Parallel CPU processing
- **psutil**: System information collection
- **rich**: Progress bars, panels, and tables

---

## Francais

### Apercu

Un outil complet de benchmark de performance systeme concu specifiquement pour les **charges de travail Data Science**.
Cet outil mesure les performances sur plusieurs dimensions : CPU (simple/multi-coeur), RAM, E/S disque, GPU, et charges
combinees CPU+GPU.

Chaque benchmark tourne pendant une **duree configurable** et rapporte un **score de debit (ops/s)** pour une
comparaison facile entre machines. La progression est affichee avec des barres de progression rich modernes.

### Fonctionnalites

- **Benchmarks CPU** : Taches orientees Data Science incluant :
    - Entrainement de regression lineaire par moindres carres
    - Algorithme de clustering K-Means (8 clusters, 20 iterations)
    - Feature engineering & preprocessing (PCA, features polynomiales)
    - Optimisation par descente de gradient (multi-start, 20D)

- **Benchmarks GPU** : Operations matricielles basees sur PyTorch
    - Multiplications de matrices 3 000x3 000 + sin
    - Sync GPU a chaque op pour un timing precis
    - Detection automatique des backends CUDA/MPS

- **Memoire & Stockage** : Operations NumPy a grande echelle et E/S fichiers
    - RAM : Allocation + sommation de tableaux de 380MB par op
    - Disque : Ecriture + lecture de 50MB par op (donnees pre-generees, I/O pur)

- **Interface Moderne** : Barres de progression rich avec compteur ops/s en direct, panels et tables

- **Timing Flexible** : Timeout configurable par benchmark (ops <0.1s donc le timeout est respecte precisement)

- **Logging Detaille** : Sortie JSON Lines + CSV avec infos systeme completes

### Prerequis

- **Python** : 3.14+
- **Gestionnaire de paquets** : uv (recommande)
- **OS** : Windows, macOS, Linux
- **Materiel** : CPU multi-coeur, GPU optionnel (NVIDIA CUDA ou Apple Metal)

### Demarrage Rapide

1. **Cloner et configurer** :
   ```bash
   git clone <repository-url>
   cd Benchmark
   ```

2. **Installer les dependances** :
   ```bash
   uv sync
   ```

3. **Lancer les benchmarks** :
   ```bash
   uv run python main.py
   ```

### Options CLI

```
--only BENCHMARK [...]       Lancer uniquement les benchmarks specifies
--skip BENCHMARK [...]       Ignorer les benchmarks specifies
--no-csv                     Desactiver l'export CSV
--list                       Lister les benchmarks disponibles

--cpu-single-timeout SECONDS Timeout CPU single-core (defaut: 30)
--cpu-multi-timeout SECONDS  Timeout CPU multi-core (defaut: 240)
--gpu-timeout SECONDS        Timeout GPU (defaut: 240)
--ram-timeout SECONDS        Timeout RAM (defaut: 30)
--disk-timeout SECONDS       Timeout Disk (defaut: 30)
--combined-timeout SECONDS   Timeout Combined CPU+GPU (defaut: 240)
```

### Durees par Defaut

| Benchmark        | Timeout par Defaut |
|------------------|:------------------:|
| CPU Single Core  |        30s         |
| CPU Multi Core   |        240s        |
| GPU              |        240s        |
| RAM              |        30s         |
| Disk             |        30s         |
| Combined CPU+GPU |        240s        |

### Sortie

L'outil genere :

- **Sortie console** : Barres de progression rich avec ops/s en direct, tableau recapitulatif
- **Fichiers de log** : `results/<hostname>_<cores>cores_<ram>GB_<gpu>_<timestamp>.log` (JSON Lines)
- **Fichiers CSV** : `results/<hostname>_<cores>cores_<ram>GB_<gpu>_<timestamp>.csv`

### Dependances

- **NumPy** : Operations mathematiques et tableaux
- **PyTorch** : Calculs GPU avec support CUDA/MPS
- **joblib** : Traitement CPU parallele
- **psutil** : Collecte d'informations systeme
- **rich** : Barres de progression, panels et tables

---

## Architecture

```
Benchmark/
├── main.py                 # Entry point & orchestration
├── benchmark_cpu.py        # Data Science CPU tasks
├── benchmark_gpu.py        # PyTorch GPU operations
├── benchmark_disk_ram.py   # Memory & I/O tests
├── pyproject.toml          # Dependencies & configuration
├── CLAUDE.md               # Claude Code instructions
└── README.md               # This file
```

## Contributing | Contribution

Feel free to submit issues and pull requests to improve the benchmark suite.

N'hesitez pas a soumettre des issues et pull requests pour ameliorer la suite de benchmark.

## License | Licence

MIT License - See LICENSE file for details.

Licence MIT - Voir le fichier LICENSE pour plus de details.

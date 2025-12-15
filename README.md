# 🚀 System Performance Benchmark Tool | Outil de Benchmark de Performance Système

[English](#english) | [Français](#français)

---

## English

### 📋 Overview

A comprehensive system performance benchmarking tool designed specifically for **Data Science workloads**. This tool
measures performance across multiple dimensions: CPU (single/multi-core), RAM, Disk I/O, GPU, and combined CPU+GPU
workloads.

### 🎯 Features

- **🧠 CPU Benchmarks**: Data Science-oriented tasks including:
    - Linear regression training with least squares
    - K-Means clustering algorithm (8 clusters, 20 iterations)
    - Feature engineering & preprocessing (PCA, polynomial features)
    - Gradient descent optimization (multi-start, 20D)

- **🎮 GPU Benchmarks**: PyTorch-based matrix operations
    - 15,000×15,000 matrix multiplications
    - Trigonometric functions (sin/cos)
    - Automatic CUDA/MPS backend detection

- **💾 Memory & Storage**: Large-scale NumPy operations and file I/O
    - RAM: 2GB array operations
    - Disk: 4GB read/write tests

- **📊 Advanced Logging**: Detailed system information and results
    - Hardware specifications (CPU, RAM, GPU)
    - Timestamped performance logs
    - JSON format for easy analysis

### 🛠 Requirements

- **Python**: 3.13+
- **Package Manager**: uv (recommended)
- **OS**: Windows, macOS, Linux
- **Hardware**: Multi-core CPU, Optional GPU (NVIDIA CUDA or Apple Metal)

### ⚡ Quick Start

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
   python main.py
   ```

### 📈 Expected Performance

- **Single-core CPU**: ~5-6 minutes
- **Multi-core CPU**: ~6-7 minutes
- **GPU**: ~2-3 minutes
- **RAM**: ~30-60 seconds
- **Disk**: ~60-120 seconds

### 📋 Output

The tool generates:

- **Console output**: Real-time progress and results
- **Log files**: `benchmark_results-YYYY-MM-DD HH:MM:SS.log`
- **System info**: Hardware specs, OS details, PyTorch configuration

### 🔧 Dependencies

- **NumPy**: Mathematical operations and arrays
- **PyTorch**: GPU computations with CUDA/MPS support
- **joblib**: Parallel CPU processing
- **psutil**: System information collection

---

## Français

### 📋 Aperçu

Un outil complet de benchmark de performance système conçu spécifiquement pour les **charges de travail Data Science**.
Cet outil mesure les performances sur plusieurs dimensions : CPU (simple/multi-cœur), RAM, E/S disque, GPU, et charges
combinées CPU+GPU.

### 🎯 Fonctionnalités

- **🧠 Benchmarks CPU** : Tâches orientées Data Science incluant :
    - Entraînement de régression linéaire par moindres carrés
    - Algorithme de clustering K-Means (8 clusters, 20 itérations)
    - Feature engineering & preprocessing (PCA, features polynomiales)
    - Optimisation par descente de gradient (multi-start, 20D)

- **🎮 Benchmarks GPU** : Opérations matricielles basées sur PyTorch
    - Multiplications de matrices 15 000×15 000
    - Fonctions trigonométriques (sin/cos)
    - Détection automatique des backends CUDA/MPS

- **💾 Mémoire & Stockage** : Opérations NumPy à grande échelle et E/S fichiers
    - RAM : Opérations sur tableaux de 2GB
    - Disque : Tests de lecture/écriture 4GB

- **📊 Logging Avancé** : Informations système détaillées et résultats
    - Spécifications matérielles (CPU, RAM, GPU)
    - Logs de performance horodatés
    - Format JSON pour analyse facile

### 🛠 Prérequis

- **Python** : 3.13+
- **Gestionnaire de paquets** : uv (recommandé)
- **OS** : Windows, macOS, Linux
- **Matériel** : CPU multi-cœur, GPU optionnel (NVIDIA CUDA ou Apple Metal)

### ⚡ Démarrage Rapide

1. **Cloner et configurer** :
   ```bash
   git clone <repository-url>
   cd Benchmark
   ```

2. **Installer les dépendances** :
   ```bash
   uv sync
   ```

3. **Lancer les benchmarks** :
   ```bash
   python main.py
   ```

### 📈 Performances Attendues

- **CPU simple-cœur** : ~5-6 minutes
- **CPU multi-cœur** : ~6-7 minutes
- **GPU** : ~2-3 minutes
- **RAM** : ~30-60 secondes
- **Disque** : ~60-120 secondes

### 📋 Sortie

L'outil génère :

- **Sortie console** : Progression en temps réel et résultats
- **Fichiers de log** : `benchmark_results-YYYY-MM-DD HH:MM:SS.log`
- **Infos système** : Spécifications matérielles, détails OS, configuration PyTorch

### 🔧 Dépendances

- **NumPy** : Opérations mathématiques et tableaux
- **PyTorch** : Calculs GPU avec support CUDA/MPS
- **joblib** : Traitement CPU parallèle
- **psutil** : Collecte d'informations système

---

## 📊 Architecture

```
Benchmark/
├── main.py                 # Entry point & orchestration
├── benchmark_cpu.py        # Data Science CPU tasks
├── benchmark_gpu.py        # PyTorch GPU operations  
├── benchmark_disk_ram.py   # Memory & I/O tests
├── pyproject.toml          # Dependencies & configuration
└── README.md              # This file
```

## 🤝 Contributing | Contribution

Feel free to submit issues and pull requests to improve the benchmark suite.

N'hésitez pas à soumettre des issues et pull requests pour améliorer la suite de benchmark.

## 📝 License | Licence

MIT License - See LICENSE file for details.

Licence MIT - Voir le fichier LICENSE pour plus de détails.
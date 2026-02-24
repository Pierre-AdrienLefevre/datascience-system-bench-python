import os
import time

import numpy as np
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


def cpu_task(iterations, loops=1):
    """
    Tâche CPU intensive orientée Data Science.
    Simule les opérations typiques d'un workflow de machine learning et d'analyse de données.
    """
    total_result = 0
    
    for loop in range(loops):
        # 1. Simulation d'entraînement de modèle de régression linéaire
        sample_size = min(100000, iterations // 2)
        if sample_size > 1000:
            # Génération de dataset synthétique
            X = np.random.randn(sample_size, 10)  # 10 features
            noise = np.random.randn(sample_size) * 0.1
            y = np.sum(X * np.random.randn(10), axis=1) + noise
            
            # Calcul des coefficients par moindres carrés (CPU intensif)
            XtX = np.dot(X.T, X)
            Xty = np.dot(X.T, y)
            # Répéter plusieurs fois pour intensifier
            for _ in range(5):
                coeffs = np.linalg.solve(XtX + np.eye(10) * 0.01, Xty)
                predictions = np.dot(X, coeffs)
                mse = np.mean((y - predictions) ** 2)
            regression_result = np.sum(coeffs) + mse
        else:
            regression_result = 0
        
        # 2. Simulation de clustering K-Means (CPU intensif)
        n_samples = min(50000, iterations // 4)
        if n_samples > 500:
            data = np.random.randn(n_samples, 5)
            centroids = np.random.randn(8, 5)  # 8 clusters
            
            # Plusieurs itérations de K-means
            for iter in range(20):  # 20 itérations pour être CPU intensif
                # Calcul des distances (très CPU intensif)
                distances = np.zeros((n_samples, 8))
                for i in range(8):
                    diff = data - centroids[i]
                    distances[:, i] = np.sum(diff ** 2, axis=1)
                
                # Assignment et update des centroids
                labels = np.argmin(distances, axis=1)
                for k in range(8):
                    mask = labels == k
                    if np.sum(mask) > 0:
                        centroids[k] = np.mean(data[mask], axis=0)
            
            kmeans_result = np.sum(centroids)
        else:
            kmeans_result = 0
        
        # 3. Simulation de preprocessing de données (feature engineering)
        array_size = min(200000, iterations // 3)
        if array_size > 1000:
            # Dataset multi-dimensionnel
            raw_data = np.random.randn(array_size, 15)
            
            # Feature engineering intensif
            # Standardisation (z-score)
            features_std = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
            
            # Création de features polynomiales (CPU intensif)
            poly_features = np.zeros((array_size, 30))
            for i in range(15):
                for j in range(i, 15):
                    poly_features[:, i+j] = features_std[:, i] * features_std[:, j]
            
            # PCA simulation (CPU très intensif)
            cov_matrix = np.cov(poly_features.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Feature selection basée sur la variance
            variances = np.var(poly_features, axis=0)
            selected_features = poly_features[:, variances > np.percentile(variances, 75)]
            
            preprocessing_result = np.sum(selected_features) + np.sum(eigenvals)
        else:
            preprocessing_result = 0
            
        # 4. Simulation d'algorithmes d'optimisation (gradient descent)
        if iterations > 1000:
            # Fonction objectif complexe avec plusieurs minima locaux
            def objective_function(x):
                return np.sum(x**4 - 16*x**2 + 5*x) + np.sum(np.sin(10*x))
            
            # Gradient descent avec multiple random starts (CPU intensif)
            best_result = float('inf')
            for start in range(min(100, iterations // 10000)):
                x = np.random.randn(20) * 5  # 20 dimensions
                learning_rate = 0.01
                
                for step in range(200):  # 200 steps de gradient descent
                    # Calcul numérique du gradient
                    grad = np.zeros_like(x)
                    h = 1e-5
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += h
                        x_minus[i] -= h
                        grad[i] = (objective_function(x_plus) - objective_function(x_minus)) / (2*h)
                    
                    x = x - learning_rate * grad
                    
                result = objective_function(x)
                best_result = min(best_result, result)
            
            optimization_result = best_result
        else:
            optimization_result = 0
            
        total_result += regression_result + kmeans_result + preprocessing_result + optimization_result

    return total_result


def cpu_benchmark_singlecore(timeout=120):
    """
    Benchmark CPU utilisant un seul cœur (séquentiel).
    Tourne pendant 'timeout' secondes et compte les opérations complétées.
    Score = ops/s (opérations par seconde).
    """
    deadline = time.time() + timeout
    completed = 0
    start = time.time()

    with Progress(
            TextColumn("[bold cyan]CPU Single Core"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[bold]{task.fields[ops]}[/bold] ops | [bold green]{task.fields[ops_s]:.2f}[/bold green] ops/s"),
    ) as progress:
        task = progress.add_task("cpu-single", total=timeout, ops=0, ops_s=0.0)
        while time.time() < deadline:
            cpu_task(10_000)
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
    }


def cpu_benchmark_multicore(timeout=120, n_jobs=None):
    """
    Benchmark multicore avec joblib utilisant tous les cœurs disponibles.
    Tourne pendant 'timeout' secondes et compte les batches complétés.
    Chaque batch lance n_jobs tâches parallèles.
    Score = ops/s (batches par seconde).
    """
    if n_jobs is None:
        n_jobs = os.cpu_count()

    deadline = time.time() + timeout
    completed = 0
    start = time.time()

    with Progress(
            TextColumn(f"[bold cyan]CPU Multi Core ({n_jobs} jobs)"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn(
                "[bold]{task.fields[ops]}[/bold] batches | [bold green]{task.fields[ops_s]:.2f}[/bold green] ops/s"),
    ) as progress:
        task = progress.add_task("cpu-multi", total=timeout, ops=0, ops_s=0.0)
        while time.time() < deadline:
            Parallel(n_jobs=n_jobs)(
                delayed(cpu_task)(10_000) for _ in range(n_jobs)
            )
            completed += 1
            elapsed = time.time() - start
            progress.update(task, completed=min(elapsed, timeout), ops=completed, ops_s=completed / elapsed)

    duration = time.time() - start
    ops_per_sec = completed / duration if duration > 0 else 0
    total_tasks = completed * n_jobs

    return {
        "duration_seconds": round(duration, 2),
        "timeout_seconds": timeout,
        "iterations_completed": completed,
        "ops_per_sec": round(ops_per_sec, 4),
        "n_jobs": n_jobs,
        "total_tasks": total_tasks,
    }

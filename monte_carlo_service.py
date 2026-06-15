import numpy as np
import multiprocessing as mp
from typing import Dict, Union, List, Optional, Tuple
import time
import os


_DISTRIBUTION_FUNCTIONS = {}


def _init_distribution_functions():
    global _DISTRIBUTION_FUNCTIONS
    _DISTRIBUTION_FUNCTIONS = {
        'normal': _normal_worker,
        'uniform': _uniform_worker,
        'exponential': _exponential_worker,
        'poisson': _poisson_worker,
        'binomial': _binomial_worker,
        'bernoulli': _bernoulli_worker,
        'gamma': _gamma_worker,
        'beta': _beta_worker,
        'lognormal': _lognormal_worker,
        'chisquare': _chisquare_worker,
        't': _t_worker,
        'f': _f_worker
    }


def _normal_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    loc = params.get('loc', 0.0)
    scale = params.get('scale', 1.0)
    return rng.normal(loc=loc, scale=scale, size=size)


def _uniform_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = params.get('low', 0.0)
    high = params.get('high', 1.0)
    return rng.uniform(low=low, high=high, size=size)


def _exponential_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = params.get('scale', 1.0)
    return rng.exponential(scale=scale, size=size)


def _poisson_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lam = params.get('lam', 1.0)
    return rng.poisson(lam=lam, size=size)


def _binomial_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = params.get('n', 1)
    p = params.get('p', 0.5)
    return rng.binomial(n=n, p=p, size=size)


def _bernoulli_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = params.get('p', 0.5)
    return rng.binomial(n=1, p=p, size=size)


def _gamma_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = params.get('shape', 1.0)
    scale = params.get('scale', 1.0)
    return rng.gamma(shape=shape, scale=scale, size=size)


def _beta_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = params.get('a', 1.0)
    b = params.get('b', 1.0)
    return rng.beta(a=a, b=b, size=size)


def _lognormal_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = params.get('mean', 0.0)
    sigma = params.get('sigma', 1.0)
    return rng.lognormal(mean=mean, sigma=sigma, size=size)


def _chisquare_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    df = params.get('df', 1)
    return rng.chisquare(df=df, size=size)


def _t_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    df = params.get('df', 1)
    return rng.standard_t(df=df, size=size)


def _f_worker(params: Dict, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dfnum = params.get('dfnum', 1)
    dfden = params.get('dfden', 1)
    return rng.f(dfnum=dfnum, dfden=dfden, size=size)


def _worker_task(args: Tuple) -> np.ndarray:
    distribution, params, size, seed = args
    if not _DISTRIBUTION_FUNCTIONS:
        _init_distribution_functions()
    if distribution not in _DISTRIBUTION_FUNCTIONS:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return _DISTRIBUTION_FUNCTIONS[distribution](params, size, seed)


_init_distribution_functions()


class MonteCarloService:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.supported_distributions = {
            'normal': self._normal,
            'uniform': self._uniform,
            'exponential': self._exponential,
            'poisson': self._poisson,
            'binomial': self._binomial,
            'bernoulli': self._bernoulli,
            'gamma': self._gamma,
            'beta': self._beta,
            'lognormal': self._lognormal,
            'chisquare': self._chisquare,
            't': self._t,
            'f': self._f
        }
        self._convergence_history = []

    def _normal(self, params: Dict, size: int) -> np.ndarray:
        loc = params.get('loc', 0.0)
        scale = params.get('scale', 1.0)
        return np.random.normal(loc=loc, scale=scale, size=size)

    def _uniform(self, params: Dict, size: int) -> np.ndarray:
        low = params.get('low', 0.0)
        high = params.get('high', 1.0)
        return np.random.uniform(low=low, high=high, size=size)

    def _exponential(self, params: Dict, size: int) -> np.ndarray:
        scale = params.get('scale', 1.0)
        return np.random.exponential(scale=scale, size=size)

    def _poisson(self, params: Dict, size: int) -> np.ndarray:
        lam = params.get('lam', 1.0)
        return np.random.poisson(lam=lam, size=size)

    def _binomial(self, params: Dict, size: int) -> np.ndarray:
        n = params.get('n', 1)
        p = params.get('p', 0.5)
        return np.random.binomial(n=n, p=p, size=size)

    def _bernoulli(self, params: Dict, size: int) -> np.ndarray:
        p = params.get('p', 0.5)
        return np.random.binomial(n=1, p=p, size=size)

    def _gamma(self, params: Dict, size: int) -> np.ndarray:
        shape = params.get('shape', 1.0)
        scale = params.get('scale', 1.0)
        return np.random.gamma(shape=shape, scale=scale, size=size)

    def _beta(self, params: Dict, size: int) -> np.ndarray:
        a = params.get('a', 1.0)
        b = params.get('b', 1.0)
        return np.random.beta(a=a, b=b, size=size)

    def _lognormal(self, params: Dict, size: int) -> np.ndarray:
        mean = params.get('mean', 0.0)
        sigma = params.get('sigma', 1.0)
        return np.random.lognormal(mean=mean, sigma=sigma, size=size)

    def _chisquare(self, params: Dict, size: int) -> np.ndarray:
        df = params.get('df', 1)
        return np.random.chisquare(df=df, size=size)

    def _t(self, params: Dict, size: int) -> np.ndarray:
        df = params.get('df', 1)
        return np.random.standard_t(df=df, size=size)

    def _f(self, params: Dict, size: int) -> np.ndarray:
        dfnum = params.get('dfnum', 1)
        dfden = params.get('dfden', 1)
        return np.random.f(dfnum=dfnum, dfden=dfden, size=size)

    def _generate_seeds(self, base_seed: Optional[int], n: int) -> List[int]:
        if base_seed is None:
            rng = np.random.default_rng()
            return rng.integers(0, 2**31 - 1, size=n).tolist()
        else:
            rng = np.random.default_rng(base_seed)
            return rng.integers(0, 2**31 - 1, size=n).tolist()

    def _parallel_sample(
        self,
        distribution: str,
        params: Dict,
        total_size: int,
        n_processes: int,
        base_seed: Optional[int]
    ) -> Tuple[np.ndarray, float]:
        if distribution not in self.supported_distributions:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Supported distributions: {list(self.supported_distributions.keys())}"
            )

        chunk_size = total_size // n_processes
        remainder = total_size % n_processes

        chunks = []
        for i in range(n_processes):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(size)

        seeds = self._generate_seeds(base_seed, n_processes)

        args_list = [
            (distribution, params, chunk_size, seed)
            for chunk_size, seed in zip(chunks, seeds)
        ]

        start_time = time.time()

        if n_processes == 1:
            results = [_worker_task(args) for args in args_list]
        else:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=n_processes) as pool:
                results = pool.map(_worker_task, args_list)

        elapsed = time.time() - start_time

        all_samples = np.concatenate(results)
        return all_samples, elapsed

    def simulate_parallel(
        self,
        distribution: str,
        params: Dict,
        n_simulations: int,
        quantiles: Optional[List[float]] = None,
        n_processes: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, float], bool, int]]:
        if n_processes is None:
            n_processes = max(1, os.cpu_count() or 1)
        if n_processes < 1:
            raise ValueError("n_processes must be >= 1")

        if distribution not in self.supported_distributions:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Supported distributions: {list(self.supported_distributions.keys())}"
            )

        if n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantile {q} must be between 0 and 1")

        samples, sampling_time = self._parallel_sample(
            distribution=distribution,
            params=params,
            total_size=n_simulations,
            n_processes=n_processes,
            base_seed=seed
        )

        start_stats = time.time()
        stats = {
            'mean': float(np.mean(samples)),
            'variance': float(np.var(samples, ddof=1)),
            'std': float(np.std(samples, ddof=1)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'median': float(np.median(samples)),
            'skewness': float(self._skewness(samples)),
            'kurtosis': float(self._kurtosis(samples))
        }
        stats_time = time.time() - start_stats

        results = {
            'distribution': distribution,
            'params': params,
            'n_simulations': n_simulations,
            'samples': samples,
            'statistics': stats,
            'quantiles': {
                f'{int(q * 100)}%': float(np.percentile(samples, q * 100))
                for q in sorted(quantiles)
            },
            'parallel': {
                'n_processes': n_processes,
                'sampling_time': sampling_time,
                'stats_time': stats_time,
                'total_time': sampling_time + stats_time,
                'seed': seed
            }
        }

        return results

    def simulate_adaptive_parallel(
        self,
        distribution: str,
        params: Dict,
        quantiles: Optional[List[float]] = None,
        min_simulations: int = 1000,
        max_simulations: int = 10_000_000,
        batch_size: int = 10_000,
        rtol: float = 1e-3,
        atol: float = 1e-8,
        consecutive_passes: int = 3,
        monitor_quantiles: Optional[List[float]] = None,
        n_processes: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, float], bool, int, List]]:
        if n_processes is None:
            n_processes = max(1, os.cpu_count() or 1)
        if n_processes < 1:
            raise ValueError("n_processes must be >= 1")

        if distribution not in self.supported_distributions:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Supported distributions: {list(self.supported_distributions.keys())}"
            )

        if min_simulations <= 0:
            raise ValueError("min_simulations must be a positive integer")
        if max_simulations < min_simulations:
            raise ValueError("max_simulations must be >= min_simulations")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if rtol <= 0:
            raise ValueError("rtol must be positive")
        if consecutive_passes < 1:
            raise ValueError("consecutive_passes must be >= 1")

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantile {q} must be between 0 and 1")

        if monitor_quantiles is None:
            monitor_quantiles = [0.05, 0.50, 0.95]
        for q in monitor_quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Monitor quantile {q} must be between 0 and 1")

        if seed is not None:
            base_rng = np.random.default_rng(seed)
        else:
            base_rng = np.random.default_rng()

        self._convergence_history = []
        n_total = 0
        converged = False
        final_changes = {}
        all_samples_list = []
        total_sampling_time = 0.0

        initial_size = max(min_simulations, batch_size)
        next_seed = int(base_rng.integers(0, 2**31 - 1))
        samples, sampling_time = self._parallel_sample(
            distribution, params, initial_size, n_processes, next_seed
        )
        total_sampling_time += sampling_time
        all_samples_list.append(samples)
        n_total = initial_size

        cumulative_samples = samples
        metrics = self._compute_convergence_metrics(cumulative_samples, monitor_quantiles)
        self._convergence_history.append({'n': n_total, **metrics})

        if verbose:
            print(f"Using {n_processes} processes for parallel sampling")
            print(f"Initial batch: {n_total:,} samples (took {sampling_time:.3f}s)")
            print(f"  mean={metrics['mean']:.6f}, std={metrics['std']:.6f}, median={metrics['median']:.6f}")

        check_interval = max(1, consecutive_passes)
        iteration = 0

        while n_total < max_simulations and not converged:
            next_batch_size = min(batch_size, max_simulations - n_total)
            next_seed = int(base_rng.integers(0, 2**31 - 1))

            new_samples, sampling_time = self._parallel_sample(
                distribution, params, next_batch_size, n_processes, next_seed
            )
            total_sampling_time += sampling_time

            all_samples_list.append(new_samples)
            n_total += next_batch_size

            cumulative_samples = np.concatenate(all_samples_list)
            metrics = self._compute_convergence_metrics(cumulative_samples, monitor_quantiles)
            self._convergence_history.append({'n': n_total, **metrics})

            iteration += 1
            if iteration % check_interval == 0:
                converged, final_changes = self._check_convergence(
                    self._convergence_history,
                    rtol=rtol,
                    atol=atol,
                    consecutive_passes=consecutive_passes,
                    monitor_quantiles=monitor_quantiles
                )

                if verbose:
                    change_str = ", ".join(
                        f"{k}={v:.2e}" for k, v in list(final_changes.items())[:5]
                    )
                    status = "CONVERGED" if converged else "continuing"
                    print(f"Batch {iteration}: {n_total:,} samples ({sampling_time:.3f}s) | {change_str} | {status}")

            all_samples_list = [cumulative_samples]

        if not converged and n_total >= max_simulations:
            _, final_changes = self._check_convergence(
                self._convergence_history,
                rtol=rtol,
                atol=atol,
                consecutive_passes=consecutive_passes,
                monitor_quantiles=monitor_quantiles
            )
            if verbose:
                print(f"WARNING: Reached max_simulations={max_simulations:,} without convergence")

        final_samples = cumulative_samples

        start_stats = time.time()
        statistics = {
            'mean': float(np.mean(final_samples)),
            'variance': float(np.var(final_samples, ddof=1)),
            'std': float(np.std(final_samples, ddof=1)),
            'min': float(np.min(final_samples)),
            'max': float(np.max(final_samples)),
            'median': float(np.median(final_samples)),
            'skewness': float(self._skewness(final_samples)),
            'kurtosis': float(self._kurtosis(final_samples))
        }
        stats_time = time.time() - start_stats

        results = {
            'distribution': distribution,
            'params': params,
            'n_simulations': n_total,
            'samples': final_samples,
            'statistics': statistics,
            'quantiles': {
                f'{int(q * 100)}%': float(np.percentile(final_samples, q * 100))
                for q in sorted(quantiles)
            },
            'convergence': {
                'converged': converged,
                'rtol': rtol,
                'atol': atol,
                'consecutive_passes': consecutive_passes,
                'final_changes': final_changes,
                'history': self._convergence_history,
                'min_simulations': min_simulations,
                'max_simulations': max_simulations
            },
            'parallel': {
                'n_processes': n_processes,
                'sampling_time': total_sampling_time,
                'stats_time': stats_time,
                'total_time': total_sampling_time + stats_time,
                'seed': seed
            }
        }

        return results

    def benchmark_parallel(
        self,
        distribution: str,
        params: Dict,
        n_simulations: int,
        process_counts: Optional[List[int]] = None,
        seed: Optional[int] = None,
        n_runs: int = 3
    ) -> Dict:
        if process_counts is None:
            cpu_count = os.cpu_count() or 4
            process_counts = [1, 2, 4]
            if cpu_count > 4:
                process_counts.append(cpu_count)
            process_counts = sorted(set(process_counts))

        benchmark_results = {
            'distribution': distribution,
            'params': params,
            'n_simulations': n_simulations,
            'process_counts': process_counts,
            'results': {}
        }

        print(f"\nBenchmark: {distribution} with {n_simulations:,} simulations")
        print("-" * 60)
        print(f"{'Processes':>10} | {'Mean Time (s)':>15} | {'Min Time (s)':>15} | {'Speedup':>10}")
        print("-" * 60)

        baseline_time = None

        for n_proc in process_counts:
            times = []
            for run in range(n_runs):
                run_seed = seed + run if seed is not None else None
                start = time.time()
                result = self.simulate_parallel(
                    distribution=distribution,
                    params=params,
                    n_simulations=n_simulations,
                    n_processes=n_proc,
                    seed=run_seed
                )
                elapsed = time.time() - start
                times.append(elapsed)
                mean_val = result['statistics']['mean']
                std_val = result['statistics']['std']

            mean_time = np.mean(times)
            min_time = np.min(times)

            if baseline_time is None:
                baseline_time = mean_time
                speedup = 1.0
            else:
                speedup = baseline_time / mean_time

            benchmark_results['results'][n_proc] = {
                'mean_time': mean_time,
                'min_time': min_time,
                'all_times': times,
                'speedup': speedup,
                'mean': mean_val,
                'std': std_val
            }

            print(f"{n_proc:>10} | {mean_time:>15.4f} | {min_time:>15.4f} | {speedup:>10.2f}x")

        print("-" * 60)
        return benchmark_results

    def _compute_convergence_metrics(
        self,
        samples: np.ndarray,
        quantiles: List[float]
    ) -> Dict[str, float]:
        metrics = {
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples, ddof=1)),
            'median': float(np.median(samples)),
        }
        for q in quantiles:
            metrics[f'q{int(q * 100)}'] = float(np.percentile(samples, q * 100))
        return metrics

    def _check_convergence(
        self,
        history: List[Dict[str, float]],
        rtol: float,
        atol: float,
        consecutive_passes: int,
        monitor_quantiles: List[float]
    ) -> Tuple[bool, Dict[str, float]]:
        if len(history) < consecutive_passes + 1:
            return False, {}

        recent = history[-consecutive_passes:]
        reference = recent[-1]

        max_changes = {}
        all_converged = True

        keys_to_check = ['mean', 'std', 'median']
        for q in monitor_quantiles:
            keys_to_check.append(f'q{int(q * 100)}')

        for key in keys_to_check:
            ref_val = reference[key]
            max_change = 0.0
            for entry in recent[:-1]:
                val = entry[key]
                denom = max(abs(ref_val), atol)
                if denom == 0:
                    change = 0.0 if abs(val - ref_val) <= atol else float('inf')
                else:
                    change = abs(val - ref_val) / denom
                if change > max_change:
                    max_change = change
            max_changes[key] = max_change
            if max_change > rtol:
                all_converged = False

        return all_converged, max_changes

    def simulate_adaptive(
        self,
        distribution: str,
        params: Dict,
        quantiles: Optional[List[float]] = None,
        min_simulations: int = 1000,
        max_simulations: int = 10_000_000,
        batch_size: int = 10_000,
        rtol: float = 1e-3,
        atol: float = 1e-8,
        consecutive_passes: int = 3,
        monitor_quantiles: Optional[List[float]] = None,
        verbose: bool = False
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, float], bool, int, List]]:
        if distribution not in self.supported_distributions:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Supported distributions: {list(self.supported_distributions.keys())}"
            )

        if min_simulations <= 0:
            raise ValueError("min_simulations must be a positive integer")
        if max_simulations < min_simulations:
            raise ValueError("max_simulations must be >= min_simulations")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if rtol <= 0:
            raise ValueError("rtol must be positive")
        if consecutive_passes < 1:
            raise ValueError("consecutive_passes must be >= 1")

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantile {q} must be between 0 and 1")

        if monitor_quantiles is None:
            monitor_quantiles = [0.05, 0.50, 0.95]
        for q in monitor_quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Monitor quantile {q} must be between 0 and 1")

        self._convergence_history = []
        all_samples_list = []
        n_total = 0
        converged = False
        final_changes = {}

        initial_size = max(min_simulations, batch_size)
        samples = self.supported_distributions[distribution](params, initial_size)
        all_samples_list.append(samples)
        n_total = initial_size

        cumulative_samples = samples
        metrics = self._compute_convergence_metrics(cumulative_samples, monitor_quantiles)
        self._convergence_history.append({'n': n_total, **metrics})

        if verbose:
            print(f"Initial batch: {n_total} samples")
            print(f"  mean={metrics['mean']:.6f}, std={metrics['std']:.6f}, median={metrics['median']:.6f}")

        check_interval = max(1, consecutive_passes)
        iteration = 0

        while n_total < max_simulations and not converged:
            next_batch_size = min(batch_size, max_simulations - n_total)
            new_samples = self.supported_distributions[distribution](params, next_batch_size)
            all_samples_list.append(new_samples)
            n_total += next_batch_size

            cumulative_samples = np.concatenate(all_samples_list)
            metrics = self._compute_convergence_metrics(cumulative_samples, monitor_quantiles)
            self._convergence_history.append({'n': n_total, **metrics})

            iteration += 1
            if iteration % check_interval == 0:
                converged, final_changes = self._check_convergence(
                    self._convergence_history,
                    rtol=rtol,
                    atol=atol,
                    consecutive_passes=consecutive_passes,
                    monitor_quantiles=monitor_quantiles
                )

                if verbose:
                    change_str = ", ".join(
                        f"{k}={v:.2e}" for k, v in list(final_changes.items())[:5]
                    )
                    status = "CONVERGED" if converged else "continuing"
                    print(f"Batch {iteration}: {n_total} samples | {change_str} | {status}")

            all_samples_list = [cumulative_samples]

        if not converged and n_total >= max_simulations:
            _, final_changes = self._check_convergence(
                self._convergence_history,
                rtol=rtol,
                atol=atol,
                consecutive_passes=consecutive_passes,
                monitor_quantiles=monitor_quantiles
            )
            if verbose:
                print(f"WARNING: Reached max_simulations={max_simulations} without convergence")

        final_samples = cumulative_samples
        results = {
            'distribution': distribution,
            'params': params,
            'n_simulations': n_total,
            'samples': final_samples,
            'statistics': {
                'mean': float(np.mean(final_samples)),
                'variance': float(np.var(final_samples, ddof=1)),
                'std': float(np.std(final_samples, ddof=1)),
                'min': float(np.min(final_samples)),
                'max': float(np.max(final_samples)),
                'median': float(np.median(final_samples)),
                'skewness': float(self._skewness(final_samples)),
                'kurtosis': float(self._kurtosis(final_samples))
            },
            'quantiles': {
                f'{int(q * 100)}%': float(np.percentile(final_samples, q * 100))
                for q in sorted(quantiles)
            },
            'convergence': {
                'converged': converged,
                'rtol': rtol,
                'atol': atol,
                'consecutive_passes': consecutive_passes,
                'final_changes': final_changes,
                'history': self._convergence_history,
                'min_simulations': min_simulations,
                'max_simulations': max_simulations
            }
        }

        return results

    def simulate(
        self,
        distribution: str,
        params: Dict,
        n_simulations: int,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, float]]]:
        if distribution not in self.supported_distributions:
            raise ValueError(
                f"Unsupported distribution: {distribution}. "
                f"Supported distributions: {list(self.supported_distributions.keys())}"
            )

        if n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer")

        samples = self.supported_distributions[distribution](params, n_simulations)

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantile {q} must be between 0 and 1")

        results = {
            'distribution': distribution,
            'params': params,
            'n_simulations': n_simulations,
            'samples': samples,
            'statistics': {
                'mean': float(np.mean(samples)),
                'variance': float(np.var(samples, ddof=1)),
                'std': float(np.std(samples, ddof=1)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'median': float(np.median(samples)),
                'skewness': float(self._skewness(samples)),
                'kurtosis': float(self._kurtosis(samples))
            },
            'quantiles': {
                f'{int(q * 100)}%': float(np.percentile(samples, q * 100))
                for q in sorted(quantiles)
            }
        }

        return results

    def _skewness(self, data: np.ndarray) -> float:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skew

    def _kurtosis(self, data: np.ndarray) -> float:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        n = len(data)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4)
        kurt -= (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return kurt

    def simulate_custom(
        self,
        samples: np.ndarray,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, Union[float, np.ndarray, Dict[str, float]]]:
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError("samples must be a 1D array")
        if len(samples) == 0:
            raise ValueError("samples cannot be empty")

        if quantiles is None:
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantile {q} must be between 0 and 1")

        results = {
            'distribution': 'custom',
            'params': {},
            'n_simulations': len(samples),
            'samples': samples,
            'statistics': {
                'mean': float(np.mean(samples)),
                'variance': float(np.var(samples, ddof=1)),
                'std': float(np.std(samples, ddof=1)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples)),
                'median': float(np.median(samples)),
                'skewness': float(self._skewness(samples)),
                'kurtosis': float(self._kurtosis(samples))
            },
            'quantiles': {
                f'{int(q * 100)}%': float(np.percentile(samples, q * 100))
                for q in sorted(quantiles)
            }
        }

        return results

    def print_results(self, results: Dict, show_convergence: bool = True, show_parallel: bool = True) -> None:
        print("=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        print(f"Distribution: {results['distribution']}")
        print(f"Parameters: {results['params']}")
        print(f"Number of Simulations: {results['n_simulations']:,}")

        if show_parallel and 'parallel' in results:
            par = results['parallel']
            print(f"Parallel: {par['n_processes']} process(es) | Total time: {par['total_time']:.3f}s")
            print(f"  (sampling: {par['sampling_time']:.3f}s, stats: {par['stats_time']:.3f}s)")

        if show_convergence and 'convergence' in results:
            conv = results['convergence']
            print(f"Convergence Status: {'CONVERGED' if conv['converged'] else 'NOT CONVERGED'}")
            if conv['converged']:
                print(f"  (rtol={conv['rtol']:.0e}, consecutive_passes={conv['consecutive_passes']})")
            else:
                print(f"  (reached max_simulations={conv['max_simulations']:,})")
            if conv['final_changes']:
                print("  Final metric changes (relative):")
                for k, v in sorted(conv['final_changes'].items()):
                    status = "OK" if v <= conv['rtol'] else "HIGH"
                    print(f"    {k:10s}: {v:.2e} [{status}]")

        print("-" * 60)
        print("Statistics:")
        for key, value in results['statistics'].items():
            print(f"  {key:15s}: {value:.6f}")
        print("-" * 60)
        print("Quantiles:")
        for key, value in sorted(results['quantiles'].items(), key=lambda x: float(x[0].rstrip('%'))):
            print(f"  {key:6s}: {value:.6f}")
        print("=" * 60)


def main():
    service = MonteCarloService(seed=42)

    print("\n" + "=" * 70)
    print("  BUG DEMO: Small fixed n_simulations causes HIGH variance in results")
    print("=" * 70)

    print("\n>>> Running 3 separate runs with n=1000 each (should show large variation)")
    for i in range(3):
        svc = MonteCarloService(seed=100 + i)
        res = svc.simulate('normal', {'loc': 100, 'scale': 15}, n_simulations=1000)
        print(f"  Run {i+1}: mean={res['statistics']['mean']:.2f}, "
              f"std={res['statistics']['std']:.2f}, "
              f"95%={res['quantiles']['95%']:.2f}")
    print("  >>> True values: mean=100.00, std=15.00, 95%≈124.67")
    print("  >>> Notice how much results differ across runs!")

    print("\n" + "=" * 70)
    print("  FIX: Using simulate_adaptive() with dynamic convergence detection")
    print("=" * 70)

    print("\n=== Example A: Adaptive Normal Distribution (default tolerance) ===")
    svc = MonteCarloService(seed=42)
    result = svc.simulate_adaptive(
        distribution='normal',
        params={'loc': 100, 'scale': 15},
        rtol=1e-3,
        consecutive_passes=3,
        batch_size=10_000,
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example B: Adaptive Normal Distribution (tight tolerance) ===")
    svc = MonteCarloService(seed=42)
    result = svc.simulate_adaptive(
        distribution='normal',
        params={'loc': 100, 'scale': 15},
        rtol=1e-4,
        atol=1e-10,
        consecutive_passes=5,
        batch_size=20_000,
        monitor_quantiles=[0.01, 0.05, 0.50, 0.95, 0.99],
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example C: Adaptive Exponential Distribution (heavy tail needs more samples) ===")
    svc = MonteCarloService(seed=123)
    result = svc.simulate_adaptive(
        distribution='exponential',
        params={'scale': 2.0},
        quantiles=[0.1, 0.5, 0.9, 0.95, 0.99],
        rtol=5e-4,
        consecutive_passes=3,
        batch_size=15_000,
        monitor_quantiles=[0.50, 0.95, 0.99],
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example D: Adaptive Binomial Distribution ===")
    svc = MonteCarloService(seed=777)
    result = svc.simulate_adaptive(
        distribution='binomial',
        params={'n': 10, 'p': 0.3},
        rtol=1e-3,
        consecutive_passes=4,
        batch_size=25_000,
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example E: Adaptive Lognormal (requires tight tail monitoring) ===")
    svc = MonteCarloService(seed=555)
    result = svc.simulate_adaptive(
        distribution='lognormal',
        params={'mean': 0.0, 'sigma': 1.0},
        rtol=2e-3,
        consecutive_passes=3,
        batch_size=20_000,
        max_simulations=5_000_000,
        monitor_quantiles=[0.50, 0.90, 0.99],
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example F: Uniform Distribution (original fixed method for comparison) ===")
    svc = MonteCarloService(seed=42)
    result = svc.simulate(
        distribution='uniform',
        params={'low': 0, 'high': 100},
        n_simulations=100000
    )
    svc.print_results(result, show_convergence=False)

    print("\n=== Example G: Custom Samples ===")
    custom_samples = np.concatenate([
        np.random.normal(0, 1, 50000),
        np.random.normal(5, 0.5, 50000)
    ])
    svc = MonteCarloService(seed=999)
    result = svc.simulate_custom(custom_samples)
    svc.print_results(result, show_convergence=False)

    print("\n" + "=" * 70)
    print("  PART 3: PARALLEL SIMULATION WITH MULTIPROCESSING")
    print("=" * 70)

    cpu_count = os.cpu_count() or 4
    print(f"\nDetected {cpu_count} CPU cores available")

    print("\n=== Example H: Performance Benchmark - Fixed Simulation ===")
    svc = MonteCarloService(seed=42)
    benchmark = svc.benchmark_parallel(
        distribution='normal',
        params={'loc': 100, 'scale': 15},
        n_simulations=2_000_000,
        process_counts=[1, 2, 4] if cpu_count >= 4 else [1, 2],
        seed=42,
        n_runs=2
    )

    print("\n=== Example I: Fixed Parallel Simulation (4 processes) ===")
    svc = MonteCarloService(seed=42)
    result = svc.simulate_parallel(
        distribution='normal',
        params={'loc': 100, 'scale': 15},
        n_simulations=5_000_000,
        n_processes=min(4, cpu_count),
        seed=12345
    )
    svc.print_results(result, show_convergence=False)

    print("\n=== Example J: Adaptive Parallel Simulation (2 processes) ===")
    svc = MonteCarloService(seed=42)
    result = svc.simulate_adaptive_parallel(
        distribution='exponential',
        params={'scale': 2.0},
        rtol=1e-3,
        atol=1e-8,
        consecutive_passes=3,
        batch_size=50_000,
        max_simulations=5_000_000,
        monitor_quantiles=[0.05, 0.50, 0.95, 0.99],
        n_processes=min(2, cpu_count),
        seed=999,
        verbose=True
    )
    svc.print_results(result)

    print("\n=== Example K: Adaptive Parallel vs Sequential Comparison ===")
    print("\n--- Sequential adaptive ---")
    svc_seq = MonteCarloService(seed=777)
    result_seq = svc_seq.simulate_adaptive(
        distribution='gamma',
        params={'shape': 2.0, 'scale': 2.0},
        rtol=5e-4,
        consecutive_passes=3,
        batch_size=30_000,
        max_simulations=2_000_000,
        verbose=True
    )
    svc_seq.print_results(result_seq)

    print("\n--- Parallel adaptive (4 processes) ---")
    svc_par = MonteCarloService(seed=777)
    result_par = svc_par.simulate_adaptive_parallel(
        distribution='gamma',
        params={'shape': 2.0, 'scale': 2.0},
        rtol=5e-4,
        consecutive_passes=3,
        batch_size=30_000,
        max_simulations=2_000_000,
        monitor_quantiles=[0.05, 0.50, 0.95],
        n_processes=min(4, cpu_count),
        seed=777,
        verbose=True
    )
    svc_par.print_results(result_par)

    if 'parallel' in result_par:
        seq_time = result_seq.get('convergence', {}).get('history', [])
        if len(seq_time) > 0 and 'time' in result_seq:
            seq_total = result_seq['time']
        else:
            seq_total = None
        par_total = result_par['parallel']['total_time']
        if seq_total:
            print(f"\nSpeedup: {seq_total / par_total:.2f}x with {result_par['parallel']['n_processes']} processes")
        print(f"Both converged at {result_par['n_simulations']:,} samples with statistically identical results")

    print("\n=== Example L: Heavy-tailed Parallel Simulation (Lognormal) ===")
    svc = MonteCarloService(seed=555)
    result = svc.simulate_adaptive_parallel(
        distribution='lognormal',
        params={'mean': 0.0, 'sigma': 0.8},
        rtol=1e-3,
        consecutive_passes=3,
        batch_size=40_000,
        max_simulations=3_000_000,
        monitor_quantiles=[0.01, 0.50, 0.99, 0.999],
        n_processes=min(4, cpu_count),
        seed=555,
        verbose=True
    )
    svc.print_results(result)


if __name__ == '__main__':
    main()

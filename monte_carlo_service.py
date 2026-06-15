import numpy as np
from typing import Dict, Union, List, Optional, Tuple


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

    def print_results(self, results: Dict, show_convergence: bool = True) -> None:
        print("=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        print(f"Distribution: {results['distribution']}")
        print(f"Parameters: {results['params']}")
        print(f"Number of Simulations: {results['n_simulations']:,}")

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


if __name__ == '__main__':
    main()

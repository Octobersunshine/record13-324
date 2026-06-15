import numpy as np
from typing import Dict, Union, List, Optional


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

    def print_results(self, results: Dict) -> None:
        print("=" * 60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 60)
        print(f"Distribution: {results['distribution']}")
        print(f"Parameters: {results['params']}")
        print(f"Number of Simulations: {results['n_simulations']}")
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

    print("\n=== Example 1: Normal Distribution ===")
    result = service.simulate(
        distribution='normal',
        params={'loc': 100, 'scale': 15},
        n_simulations=100000
    )
    service.print_results(result)

    print("\n=== Example 2: Uniform Distribution ===")
    result = service.simulate(
        distribution='uniform',
        params={'low': 0, 'high': 100},
        n_simulations=100000
    )
    service.print_results(result)

    print("\n=== Example 3: Exponential Distribution ===")
    result = service.simulate(
        distribution='exponential',
        params={'scale': 2.0},
        n_simulations=100000,
        quantiles=[0.1, 0.5, 0.9, 0.95, 0.99]
    )
    service.print_results(result)

    print("\n=== Example 4: Binomial Distribution ===")
    result = service.simulate(
        distribution='binomial',
        params={'n': 10, 'p': 0.3},
        n_simulations=100000
    )
    service.print_results(result)

    print("\n=== Example 5: Custom Samples ===")
    custom_samples = np.concatenate([
        np.random.normal(0, 1, 50000),
        np.random.normal(5, 0.5, 50000)
    ])
    result = service.simulate_custom(custom_samples)
    service.print_results(result)


if __name__ == '__main__':
    main()

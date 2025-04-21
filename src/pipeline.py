import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from modules.model_toa import ModelTOA
from modules.parametric_signal_selection import (
    ParametricSignalSelector,
    FrequencyGridMode,
)
from modules.metric import quality_metric


def evaluate_selector_params(
    alpha_threshold: float,
    averaging_threshold: float,
    freq_step: float,
    model_params: dict,
) -> float:
    """Вычисление метрики качества для заданных параметров"""
    model = ModelTOA(**model_params)
    model.generate_toa()
    reference_PRI = model.get_period_array()

    if len(reference_PRI) == 0:
        return np.inf

    min_freq = 1 / np.max(reference_PRI)
    max_freq = 1 / np.min(reference_PRI)

    selector = ParametricSignalSelector(
        T=model_params["T"],
        alpha_threshold=alpha_threshold,
        averaging_threshold=averaging_threshold,
        min_freq=min_freq,
        max_freq=max_freq,
        freq_step=freq_step,
    )

    try:
        estimated_PRI = selector.estimate_PRI(
            model.get_toa_array(), FrequencyGridMode.LINEAR
        )
        return quality_metric(reference_PRI, estimated_PRI)
    except:
        return np.inf


def optimize_parameter(
    model_param_name: str,
    model_param_values: list,
    fixed_model_params: dict,
    n_calls: int = 30,
) -> pd.DataFrame:
    """Байесовская оптимизация для одного параметра модели"""
    space = [
        Real(0.01, 0.1, name="alpha_threshold"),
        Real(0.05, 0.5, name="averaging_threshold"),
        Real(0.01, 0.1, name="freq_step"),
    ]

    results = []
    for param_value in model_param_values:
        current_params = fixed_model_params.copy()
        current_params[model_param_name] = param_value

        @use_named_args(space)
        def objective(**kwargs):
            return evaluate_selector_params(model_params=current_params, **kwargs)

        res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        results.append(
            {
                model_param_name: param_value,
                "alpha_threshold": res.x[0],
                "averaging_threshold": res.x[1],
                "freq_step": res.x[2],
                "metric": res.fun,
            }
        )

    return pd.DataFrame(results).sort_values("metric")


if __name__ == "__main__":
    fixed_params = {
        "T": 100,
        "std_toa": 0.1,
        "time_start_jitter": 0.5,
        "signal_loss_rate": 0.1,
        "min_period": 0.5,
        "max_period": 2.0,
        "period_num": 3,
        "period_diff_threshold": 0.1,
    }

    std_toa_results = optimize_parameter(
        "std_toa", np.linspace(0.1, 1.0, 5), fixed_params
    )

    period_num_results = optimize_parameter("period_num", [2, 3, 4, 5], fixed_params)

    signal_loss_results = optimize_parameter(
        "signal_loss_rate", np.linspace(0.0, 0.5, 5), fixed_params
    )

    print("Результаты для std_toa:")
    print(std_toa_results)

    print("\nРезультаты для period_num:")
    print(period_num_results)

    print("\nРезультаты для signal_loss_rate:")
    print(signal_loss_results)

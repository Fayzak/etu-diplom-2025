import numpy as np

from modules.model_toa import ModelTOA
from modules.parametric_signal_selection import (
    ParametricSignalSelector,
    FrequencyGridMode,
)
from modules.metric import quality_metric


if __name__ == "__main__":
    model_toa = ModelTOA(
        T=4,
        std_toa=1e-6,
        time_start_jitter=1e-2,
        signal_loss_rate=0.3,
        min_period=1e-3,
        max_period=10e-3,
        period_num=5,
        period_diff_threshold=5e-5,
    )
    model_toa.generate_toa()
    model_toa.draw_scatter_plot(0, 50)
    toa_array = model_toa.get_toa_array()
    period_array = model_toa.get_period_array()
    T = model_toa.get_T()

    parametric_signal_selector = ParametricSignalSelector(
        T=T,
        min_freq=50,
        max_freq=1000,
        alpha_threshold=5e-1,
        averaging_threshold=5e-1,
        freq_step=1e-1,
    )

    estimated_PRI = parametric_signal_selector.estimate_PRI(
        toa_array=toa_array, mode=FrequencyGridMode.LINEAR
    )
    print(estimated_PRI)
    print(np.sort(period_array))
    parametric_signal_selector.plot_signal_spectrum(
        period_array=period_array, toa_array=toa_array, mode=FrequencyGridMode.LINEAR
    )

    metric_result = quality_metric(period_array, estimated_PRI)
    print(metric_result)

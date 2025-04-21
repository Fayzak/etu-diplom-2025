from enum import Enum
import numpy as np
import matplotlib.pyplot as plt


class FrequencyGridMode(Enum):
    LINEAR = "linear"
    LOG = "log"
    NOMINAL = "nominal"


class ParametricSignalSelector:
    def __init__(
        self,
        T: int,
        alpha_threshold: float,
        averaging_threshold: float,
        min_freq: int = None,
        max_freq: int = None,
        freqs: list[float] = None,
        freq_step: float = None,
        freq_count: int = None,
    ) -> None:
        """Класс параметрической селекции сигналов.

        Args:
            T (int): Длительность пачки.
            alpha_threshold (float): Порог обнаружения PRF.
            averaging_threshold (float): Порог усреднения PRF.
            min_freq (int): Минимальная частота.
            max_freq (int): Максимальная частота.
            freqs (list[float]): Номиналы частот.
            freq_step (float): Шаг сетки.
            freq_count (int): Количество точек в сетке.
        """
        self._T = T
        self._alpha = alpha_threshold
        self._averaging_threshold = averaging_threshold
        self._grid_params = {
            "min_freq": min_freq,
            "max_freq": max_freq,
            "freqs": freqs,
            "freq_step": freq_step,
            "freq_count": freq_count,
        }

    def __generate_linear_freq(self) -> np.ndarray:
        return np.arange(
            self._grid_params["min_freq"],
            self._grid_params["max_freq"],
            self._grid_params["freq_step"],
        )

    def __generate_log_freq(self) -> np.ndarray:
        return (
            np.log(
                np.linspace(
                    np.exp(
                        self._grid_params["min_freq"] / self._grid_params["max_freq"]
                    ),
                    np.exp(1),
                    num=self._grid_params["freq_count"],
                )
            )
            * self._grid_params["max_freq"]
        )

    def __get_frequencies(self, mode: FrequencyGridMode) -> np.ndarray:
        if mode == FrequencyGridMode.LINEAR:
            return self.__generate_linear_freq()
        elif mode == FrequencyGridMode.LOG:
            return self.__generate_log_freq()
        elif mode == FrequencyGridMode.NOMINAL:
            return np.array(self._grid_params["freqs"])
        else:
            raise ValueError("Неподдерживаемый режим сетки")

    def __calculate_amplitude(
        self, frequencies: np.ndarray, toa_array: np.ndarray
    ) -> np.ndarray:
        amplitude = np.exp(1j * 2 * np.pi * frequencies * toa_array.reshape(-1, 1))
        amplitude = np.abs(np.sum(amplitude, axis=0))
        return amplitude

    def _estimate_PRF(
        self, toa_array: np.ndarray, mode: FrequencyGridMode
    ) -> np.ndarray:
        frequencies = self.__get_frequencies(mode)
        amplitude = self.__calculate_amplitude(frequencies, toa_array)

        PRF = frequencies[amplitude > (self._T * frequencies) * self._alpha]
        amp = amplitude[amplitude > (self._T * frequencies) * self._alpha]
        est_PRF = np.sort(PRF[np.argsort(amp)])
        return est_PRF

    def _average_estimated_PRF(self, estimated_PRF: np.ndarray) -> np.ndarray:
        averaged_PRF = []

        if not estimated_PRF.size > 0:
            return np.array(averaged_PRF)

        sorted_PRF = np.sort(estimated_PRF)
        current_group = [sorted_PRF[0]]

        for i in range(1, len(sorted_PRF)):
            if abs(sorted_PRF[i] - sorted_PRF[i - 1]) < self._averaging_threshold:
                current_group.append(sorted_PRF[i])
            else:
                averaged_PRF.append(np.mean(current_group))
                current_group = [sorted_PRF[i]]

        averaged_PRF.append(np.mean(current_group))

        return np.array(averaged_PRF)

    def estimate_PRI(
        self, toa_array: np.ndarray, mode: FrequencyGridMode
    ) -> np.ndarray:
        estimated_PRF = self._estimate_PRF(toa_array, mode)
        averaged_PRF = self._average_estimated_PRF(estimated_PRF)
        return np.sort(1 / averaged_PRF)

    def plot_signal_spectrum(
        self, period_array: np.ndarray, toa_array: np.ndarray, mode: FrequencyGridMode
    ) -> None:
        frequencies = self.__get_frequencies(mode)
        amplitude = self.__calculate_amplitude(frequencies, toa_array)
        estimated_PRF = self._estimate_PRF(toa_array, mode)
        averaged_PRF = self._average_estimated_PRF(estimated_PRF)

        plt.figure(figsize=(16, 5))
        plt.plot(frequencies, amplitude, label="Спектр")
        plt.plot(
            1 / period_array,
            (self._T / period_array) * self._alpha,
            "go",
            label="Истинные PRF",
        )
        plt.plot(frequencies, (self._T * frequencies) * self._alpha, "m", label="Порог")
        plt.vlines(
            averaged_PRF,
            0,
            max(amplitude) * 0.5,
            "r",
            linewidth=1.5,
            label="Обнаруженные PRF",
        )

        plt.xlim(self._grid_params["min_freq"], self._grid_params["max_freq"])
        plt.title("Спектр последовательности сигналов с обнаруженными PRF")
        plt.xlabel("Импульсная радиочастота, Гц")
        plt.ylabel("Амплитуда")
        plt.grid()
        plt.legend()
        plt.show()

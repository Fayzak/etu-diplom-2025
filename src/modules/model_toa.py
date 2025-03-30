import numpy as np
import matplotlib.pyplot as plt


class ModelTOA:
    def __init__(
        self,
        T: int,
        std_toa: float,
        time_start_jitter: float,
        signal_loss_rate: float,
        min_period: float,
        max_period: float,
        period_num: int,
        period_diff_threshold: float,
    ) -> None:
        """Класс моделирования времен прихода.

        Args:
            T (int): Длительность каждой пачки.
            std_toa (float): СКО времен прихода.
            time_start_jitter (float): Максимальный случайный сдвиг каждой пачки (равномерное распределение от 0 до этого значения).
            signal_loss_rate (float): Процент потерь сигналов.
            min_period (float): Минимальный период следования.
            max_period (float): Маскимальный период следования.
            period_num (int): Количество различных периодов следования (число источников).
            period_diff_threshold (int): Пороговое значение для разницы между периодами (разрешающая способность).
        """
        self.__T = T
        self.__std_toa = std_toa
        self.__time_start_jitter = time_start_jitter
        self.__signal_loss_rate = signal_loss_rate
        self.__period_diff_threshold = period_diff_threshold
        self.__period_array = self.__generate_unique_periods(
            min_period, max_period, period_num
        )

        self._toa_array = np.array([])
        self._labels = np.array([])
        self._sequence_lens = {}

    def __generate_unique_periods(
        self, min_period: float, max_period: float, period_num: int
    ) -> np.ndarray:
        periods = np.zeros(period_num)
        for i in range(period_num):
            while True:
                period = np.random.uniform(min_period, max_period)
                if all(abs(period - periods[:i]) > self.__period_diff_threshold):
                    periods[i] = period
                    break
        return periods

    def generate_toa(self) -> None:
        for i, period in enumerate(self.__period_array):
            time_arrival = np.arange(0, self.__T, period) + np.random.uniform(
                0, self.__time_start_jitter, 1
            )
            time_arrival += np.random.normal(0, self.__std_toa, len(time_arrival))

            mask = np.random.uniform(size=len(time_arrival)) >= self.__signal_loss_rate

            time_arrival = time_arrival[mask]
            label = np.ones(len(time_arrival)) * i
            self._labels = np.concatenate([self._labels, label])

            self._sequence_lens[i] = len(time_arrival)
            self._toa_array = np.concatenate([self._toa_array, time_arrival])

        self._labels = self._labels[np.argsort(self._toa_array)]
        self._toa_array = np.sort(self._toa_array)
        self._toa_array -= min(self._toa_array)

    def get_labels(self) -> np.ndarray:
        return self._labels

    def get_toa_array(self) -> np.ndarray:
        return self._toa_array

    def get_period_array(self) -> np.ndarray:
        return self.__period_array

    def get_std_toa(self) -> float:
        return self.__std_toa

    def get_T(self) -> int:
        return self.__T

    def get_signal_loss_rate(self) -> float:
        return self.__signal_loss_rate

    def draw_scatter_plot(self, start: int, end: int) -> None:
        plt.title("Диаграмма рассеяния времен прихода")
        plt.ylabel("Время прихода, с")
        plt.xlabel("Число отсчетов")

        unique_labels = np.unique(self._labels[start:end])
        colors = plt.get_cmap("viridis", len(unique_labels))(
            np.arange(len(unique_labels))
        )

        scatter = plt.scatter(
            x=np.arange(len(self._toa_array))[start:end],
            y=self._toa_array[start:end],
            c=self._labels[start:end],
            cmap="viridis",
        )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i],
                markersize=10,
                label=f"Period {int(label)}",
            )
            for i, label in enumerate(unique_labels)
        ]
        plt.legend(handles=legend_elements, title="Periods")

        plt.grid()
        plt.show()

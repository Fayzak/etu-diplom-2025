import numpy as np
import matplotlib.pyplot as plt


class ParametricSignalSelector:
  def __init__(self,
               min_freq: int,
               max_freq: int,
               alpha_threshold: float,
               averaging_threshold: float) -> None:
        """Класс параметрической селекции сигналов.

        Args:
            min_freq (int): Минимальная радиочастота.
            max_freq (float): Максимальная радиочастота.
            alpha_threshold (float): Порог для определения найденных радиочастот.
            averaging_threshold (float): Порог для усреднения найденных радиочастот.
        """
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._alpha = alpha_threshold
        self._averaging_threshold = averaging_threshold

  def __get_frequences(self) -> np.ndarray:
      return np.log(np.linspace(np.exp(self._min_freq/self._max_freq), np.exp(1), num = 8000)) * max_freq

  def __calculate_amplitude(self, frequencies: np.ndarray, toa_array: np.ndarray) -> np.ndarray:
      amplitude = np.exp(1j * 2 * np.pi * frequencies * toa_array.reshape(-1,1))
      amplitude = np.abs(np.sum(amplitude, axis=0))
      return amplitude

  def _estimate_PRF(self, toa_array: np.ndarray) -> np.ndarray:
      frequencies = self.__get_frequences()
      amplitude = self.__calculate_amplitude(frequencies, toa_array)

      PRF = frequencies[amplitude > (T * frequencies)*self._alpha]
      amp = amplitude[amplitude > (T * frequencies)*self._alpha]
      est_PRF = np.sort(PRF[np.argsort(amp)])
      return est_PRF

  def _average_estimated_PRF(self, estimated_PRF: np.ndarray) -> np.ndarray:
      sorted_PRF = np.sort(est_PRF)

      grouped_PRF = []
      current_group = [sorted_PRF[0]]

      for i in range(1, len(sorted_PRF)):
          if abs(sorted_PRF[i] - sorted_PRF[i - 1]) < self._averaging_threshold: # Нужен ли Махалонобис и как его применить?
              current_group.append(sorted_PRF[i])
          else:
              grouped_PRF.append(np.mean(current_group))
              current_group = [sorted_PRF[i]]

      grouped_PRF.append(np.mean(current_group))

      return np.array(grouped_PRF)

  def estimate_PRI(self, toa_array: np.ndarray) -> np.ndarray:
      estimated_PRF = self._estimate_PRF(toa_array)
      averaged_PRF = self._average_estimated_PRF(estimated_PRF)
      return np.sort(1 / averaged_PRF)

  def plot_signal_spectrum(self, toa_array: np.ndarray) -> None:
      frequencies = self.__get_frequences()
      amplitude = self.__calculate_amplitude(frequencies, toa_array)
      estimated_PRF = self._estimate_PRF(toa_array)
      averaged_PRF = self._average_estimated_PRF(estimated_PRF)

      plt.figure(figsize=(19,5))
      plt.plot(frequencies, amplitude, label='Спектр')
      plt.plot(1 / period_array, (T / period_array)*self._alpha, 'go', label='Найденные PRF')
      plt.plot(frequencies, (T * frequencies)*self._alpha, 'm', label='Порог')
      plt.vlines(averaged_PRF, 0, max(amplitude)*0.5, 'r', linewidth=1.5, label='Усреднённые PRF')

      plt.xlim(min_freq, max_freq)
      plt.title('Спектр сигнала с усредненными PRF')
      plt.xlabel('Импульсная радиочастота, Гц')
      plt.ylabel('Амплитуда')
      plt.grid()
      plt.legend()
      plt.show()

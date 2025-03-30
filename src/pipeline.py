import itertools
import numpy as np
import random
import pandas as pd

# from model import ModelTOA
# from parametric_signal_selection import ParametricSignalSelector
# from metric import quality_metric


# Параметры для вариации
param_ranges_toa = {
    "T": [2, 6, 10],
    "std_toa": [5e-7, 1e-6, 5e-6],
    "time_start_jitter": [1e-3, 5e-3, 1e-2],
    "signal_loss_rate": [0.1, 0.2, 0.3],
    "min_period": [1e-3, 3e-3, 5e-3],
    "max_period": [10e-3, 30e-3, 50e-3],
    "period_num": [2, 6, 10],
    "period_diff_threshold": [5e-5, 3e-5, 1e-5],
}

param_ranges_selector = {
    "min_freq": [30, 50, 80],
    "max_freq": [500, 1000, 1500],
    "alpha_threshold": [0.1, 0.4, 0.7],
    "averaging_threshold": [0.1, 0.4, 0.7],
}

# Генетический алгоритм
population_size = 20
generations = 5
mutation_rate = 0.1


def initialize_population(size):
    population = []
    for _ in range(size):
        params_toa = [random.choice(param_ranges_toa[key]) for key in param_ranges_toa]
        params_selector = [
            random.choice(param_ranges_selector[key]) for key in param_ranges_selector
        ]
        population.append((params_toa, params_selector))
    return population


def evaluate_fitness(individual):
    params_toa, params_selector = individual
    model_toa = ModelTOA(*params_toa)
    model_toa.generate_toa()
    toa_array = model_toa.get_toa_array()
    period_array = model_toa.get_period_array()

    parametric_signal_selector = ParametricSignalSelector(*params_selector)
    PRI = parametric_signal_selector.estimate_PRI(toa_array)

    metric_result = quality_metric(period_array, PRI)
    return metric_result


def select_parents(population, fitness):
    sorted_population = [
        ind for _, ind in sorted(zip(fitness, population), key=lambda x: x[0])
    ]
    return sorted_population[: population_size // 2]


def crossover(parents):
    offspring = []
    for _ in range(population_size):
        parent1, parent2 = random.sample(parents, 2)
        crossover_point_toa = random.randint(1, len(parent1[0]) - 1)
        crossover_point_selector = random.randint(1, len(parent1[1]) - 1)
        child_toa = parent1[0][:crossover_point_toa] + parent2[0][crossover_point_toa:]
        child_selector = (
            parent1[1][:crossover_point_selector]
            + parent2[1][crossover_point_selector:]
        )
        offspring.append((child_toa, child_selector))
    return offspring


def mutate(individual):
    if random.random() < mutation_rate:
        index_toa = random.randint(0, len(individual[0]) - 1)
        individual[0][index_toa] = random.choice(
            list(param_ranges_toa.values())[index_toa]
        )
    if random.random() < mutation_rate:
        index_selector = random.randint(0, len(individual[1]) - 1)
        individual[1][index_selector] = random.choice(
            list(param_ranges_selector.values())[index_selector]
        )


# Основной цикл генетического алгоритма
population = initialize_population(population_size)
for generation in range(generations):
    fitness = [evaluate_fitness(ind) for ind in population]
    parents = select_parents(population, fitness)
    population = crossover(parents)
    for individual in population:
        mutate(individual)

# Оценка последнего поколения
final_fitness = [evaluate_fitness(ind) for ind in population]
sorted_population = sorted(zip(final_fitness, population), key=lambda x: x[0])

# Получение 10 лучших и 10 худших результатов
best_individuals = sorted_population[:10]
worst_individuals = sorted_population[-10:]

# Создание DataFrame для лучших результатов
best_data = []
for metric, (params_toa, params_selector) in best_individuals:
    row = params_toa + params_selector + [metric]
    best_data.append(row)

best_columns = [
    "T",
    "std_toa",
    "time_start_jitter",
    "signal_loss_rate",
    "min_period",
    "max_period",
    "period_num",
    "period_diff_threshold",
    "min_freq",
    "max_freq",
    "alpha_threshold",
    "averaging_threshold",
    "Metric",
]
best_df = pd.DataFrame(best_data, columns=best_columns)

# Создание DataFrame для худших результатов
worst_data = []
for metric, (params_toa, params_selector) in worst_individuals:
    row = params_toa + params_selector + [metric]
    worst_data.append(row)

worst_df = pd.DataFrame(worst_data, columns=best_columns)

# Вывод таблиц
print("10 лучших результатов:")
print(best_df)

print("\n10 худших результатов:")
print(worst_df)

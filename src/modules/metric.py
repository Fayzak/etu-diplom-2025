import numpy as np


def quality_metric(reference_PRI: np.ndarray, estimated_PRI: np.ndarray) -> float:
    """Используется расстояние Хаусдорффа для одномерных массивов"""
    directed_hausdorff_1d = lambda set1, set2: np.max(
        [np.min([np.abs(x - y) for y in set2]) for x in set1]
    )
    d1 = directed_hausdorff_1d(reference_PRI, estimated_PRI)
    d2 = directed_hausdorff_1d(estimated_PRI, reference_PRI)
    return max(d1, d2)

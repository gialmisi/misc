import numpy as np
import pandas as pd


def color_solutions(
    objectives: np.ndarray,
    ref_point: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray = None,
):
    if nadir is not None:
        distances = {
            "L1 distance from ideal point": (1, ideal),
            "L1 distance from nadir point": (1, nadir),
            "L1 distance from reference point": (1, ref_point),
            "L2 distance from ideal point": (2, ideal),
            "L2 distance from nadir point": (2, nadir),
            "L2 distance from reference point": (2, ref_point),
            "L-inf distance from ideal point": (np.inf, ideal),
            "L-inf distance from nadir point": (np.inf, nadir),
            "L-inf distance from reference point": (np.inf, ref_point),
        }
    else: 
        distances = {
            "L1 distance from ideal point": (1, ideal),
            #  "L1 distance from nadir point": (1, nadir),
            "L1 distance from reference point": (1, ref_point),
            "L2 distance from ideal point": (2, ideal),
            #  "L2 distance from nadir point": (2, nadir),
            "L2 distance from reference point": (2, ref_point),
            "L-inf distance from ideal point": (np.inf, ideal),
            #  "L-inf distance from nadir point": (np.inf, nadir),
            "L-inf distance from reference point": (np.inf, ref_point),
        }
    color_data = pd.DataFrame(columns=distances.keys, index=range(len(objectives)))
    for key, val in distances.items():
        color_data[key] = np.linalg.norm(objectives - val[1], axis=1, ord=val[0])
    return color_data

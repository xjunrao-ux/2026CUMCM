from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, r"D:\Users\Xenop\Documents\Github\2026CUMCM\code")
import a1_coupled_fvm as model

root = Path(r"D:\Users\Xenop\Documents\Github\2026CUMCM")
environment = model.load_environment(model.find_default_data_path(root))
for dr_cm, dt_s in [(0.1, 1.0), (0.05, 0.5)]:
    result = model.simulate_moisture(
        environment.times_s,
        environment.moisture_kg_kg,
        radial_step_cm=dr_cm,
        time_step_s=dt_s,
    )
    surface = model.sample_moisture_field(
        result, np.array([1800.0]), np.array([2.0])
    )[0, 0]
    average = np.sum(result.volumes_m3_m * result.moisture_kg_kg[-1]) / np.sum(
        result.volumes_m3_m
    )
    print(dr_cm, dt_s, "surface", surface, "average", average)

DEFAULT_HMM = {
    "em_iter": 1000,
    "penalty": 0.01,
    "glm_iter": 500,
    "eps": 1e-6,
}

DEFAULT_CHANNEL = {"blue": "none", "green": "gbm", "red": "vasc"}

DEFAULT_COL = (
    "delta_t", "tme_label", "state_label", "sum_green", "growth_rate",
    "speed", "direction", "roi_speed", "sin", "cos", "roi_sin", "roi_cos",
    "roi_direction", "weighted_sin", "weighted_cos", "polarization",
    "cos_sim", "faster", "directional_change", "speed_change",
    "roi_directional_change", "roi_speed_change", "vascular_distance",
    "adMAD", "speed_treat", "direction_treat", "roi_speed_treat",
    "roi_direction_treat", "polarization_treat", "cos_sim_treat", "faster_treat",
    "directional_change_treat", "speed_change_treat", "roi_directional_change_treat",
    "roi_speed_change_treat", "growth_rate_treat", "sum_green_treat", "tme_label_treat",
    "vascular_distance_treat"
)

NON_SCALE_COLUMNS = [
    "sin",
    "cos",
    "roi_sin",
    "roi_cos",
    "roi_direction",
    "weighted_sin",
    "weighted_cos",
    "polarization",
    "cos_sim",
    "directional_change",
    "roi_directional_change",
    "tme_label",
    "direction_treat",
    "roi_direction_treat",
    "polarization_treat",
    "cos_sim_treat",
    "directional_change_treat",
    "roi_directional_change_treat",
    "tme_label_treat",
    "vascular_distance_treat",
    "is_treatment",
]

SOFTMAX_COLUMNS = [
    "Branching",
    "Diffuse translocation",
    "Junk",
    "Locomotion",
    "Perivascular translocation",
    "Round",
]

HARD_CODED_FEATURES = [
    "exp",
    "cellID",
    "roi",
    "time",
    "Branching",
    "Diffuse translocation",
    "Junk",
    "Locomotion",
    "state_label",
    "Perivascular translocation",
    "Round",
]

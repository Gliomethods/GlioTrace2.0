DEFAULT_HMM = {
    "em_iter": 1000,
    "penalty": 0.01,
    "glm_iter": 500,
    "eps": 1e-6,
}

DEFAULT_CHANNEL = {"blue": "none", "green": "gbm", "red": "vasc"}

DEFAULT_COL = (
    "tme_label_treat",          "tme_label",
    "speed_treat",              "speed",
    "polarization_treat",       "polarizaition"
    "cos_sim_treat",            "cos_sim"
    "vascular_distance_treat",  "vascular_distance",
    "sum_green", "adMAD", "is_treatment"
)

NON_SCALE_COLUMNS = [
    "polarization",
    "polarization_treat",
    "cos_sim",
    "cos_sim_treat",
    "tme_label",
    "tme_label_treat"
    "is_treatment"
]

SOFTMAX_COLUMNS = [
    "Branching",
    "Diffuse translocation",
    "Crowded",
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
    "Crowded",
    "Locomotion",
    "state_label",
    "Perivascular translocation",
    "Round",
]

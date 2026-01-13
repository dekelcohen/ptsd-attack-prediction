from enum import Enum

class Biomarker(Enum):
    Pr = "pulse-rate"
    Eda = "eda"  # Electrodermal Activity
    EdaPhasic = "eda-phasic"
    EdaTonic = "eda-tonic"
    AccStd = "accelerometers-std"  # Accelerometer Magnitude Standard Deviation
    Prv = "prv"  # Pulse Rate Variability
    Met = "met"  # Metabolic Equivalent of Task
    Temp = "temperature"  # Skin Temperature in °C
    RR = "respiratory-rate"

biomarker_value_names = {
    Biomarker.Pr: "pulse_rate_bpm",
    Biomarker.Eda: "eda_scl_usiemens",
    Biomarker.EdaPhasic: "EDA_Phasic",
    Biomarker.EdaTonic: "EDA_Tonic",
    Biomarker.AccStd: "accelerometers_std_g",
    Biomarker.Prv: "prv_rmssd_ms",
    Biomarker.Met: "met",
    Biomarker.Temp: "temperature_celsius",
    Biomarker.RR: "respiratory_rate_brpm"
}

biomarker_colors = {
    Biomarker.Pr: "crimson",
    Biomarker.Eda: "darkorange",
    Biomarker.EdaPhasic: "purple",
    Biomarker.EdaTonic: "pink",
    Biomarker.AccStd: "aqua",
    Biomarker.Prv: "blue",
    Biomarker.Met: "green",
    Biomarker.Temp: "yellow",
    Biomarker.RR: "purple"
}

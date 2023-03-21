from itertools import product
from typing import Dict, Sequence


def _derived_fe_dict():
    feature_dict = {
        "facial": {
            "group_1": {
                "raw_features": [f"fac_LMK{i:02}disp" for i in range(1, 68)]
                + [f"fac_AU{i:02}int" for i in range(1, 27) if i not in [16]]
                + ["fac_AU45int"]
                + [
                    f"fac_{em}int{intens}"
                    for em, intens in product(["hap", "sad", "sur", "fea", "ang", "dis", "con"], ["soft", "hard"])
                ]
                + [
                    # facial asymmetry
                    "fac_asymmaskmouth",
                    "fac_asymmaskeyebrow",
                    "fac_asymmaskeye",
                    "fac_asymmaskcom",
                ],
                "derived_features": ["mean", "std"],
            },
            "group_2": {
                "raw_features":
                # list comprehension
                [f"fac_AU{i:02}pres" for i in range(1, 29) if i not in [3, 8, 11, 13, 16, 18, 19, 21, 22, 24, 27, 29]]
                + ["fac_AU45pres"]
                + [
                    # emotional expressivity - pct
                    "fac_happres",
                    "fac_sadpres",
                    "fac_surpres",
                    "fac_feapres",
                    "fac_angpres",
                    "fac_dispres",
                    "fac_conpres",
                ],
                "derived_features": ["pct"],
            },
            "group_3": {
                "raw_features": [
                    # overall expressivity
                    "fac_comintsoft",
                    "fac_cominthard",
                    "fac_comlowintsoft",
                    "fac_comlowinthard",
                    "fac_comuppintsoft",
                    "fac_comuppinthard",
                    # pain expressivity
                    "fac_paiintsoft",
                    "fac_paiinthard",
                    # positive and negative, not in documentation
                    "fac_posintsoft",
                    "fac_posinthard",
                    "fac_negintsoft",
                    "fac_neginthard",
                ],
                "derived_features": ["mean", "std", "pct"],
            },
        },
        "acoustic": {
            "group_1": {
                "raw_features": [
                    # fundamental frequency
                    "aco_ff",
                    # formant frequencies
                    "aco_fm1",
                    "aco_fm2",
                    "aco_fm3",
                    "aco_fm4",
                    # audio intensity
                    "aco_int",
                    # harmonics-to-noise-ratio
                    "aco_hnr",
                ],
                "derived_features": ["mean", "std", "range"],
            },
            "group_2": {
                "raw_features": [f"aco_mfcc{i}" for i in range(1, 13)],
                "derived_features": ["mean"],
            },
        },
        "movement": {
            "group_1": {
                "raw_features": [
                    # head movement
                    "mov_headvel",
                    "mov_hposedist",
                    "mov_hposepitch",
                    "mov_hposeyaw",
                    "mov_hposeroll",
                    # eye gaze directionality
                    "mov_lefteyex",
                    "mov_lefteyey",
                    "mov_lefteyez",
                    "mov_righteyex",
                    "mov_righteyey",
                    "mov_righteyez",
                    "mov_leyedisp",
                    "mov_reyedisp",
                ],
                "derived_features": ["mean", "std"],
            },
            "group_2": {"raw_features": ["mov_eyeblink"], "derived_features": ["count", "dur_mean", "dur_std"]},
        },
        "eyeblink_ear": {"group_1": {"raw_features": ["mov_blink_ear"], "derived_features": ["mean", "std"]}},
        "acoustic_seg": {
            "group_1": {"raw_features": ["aco_jitter", "aco_shimmer", "aco_gne"], "derived_features": ["wmean", "wstd"]}
        },
        "audio_seg": {
            "group_1": {
                "raw_features": [
                    # vocal tremor
                    "mov_freqtremfreq",
                    "mov_freqtremindex",
                    "mov_freqtrempindex",
                    "mov_amptremfreq",
                    "mov_amptremindex",
                    "mov_amptrempindex",
                    # voice prevalence
                    "aco_voicepct",
                ],
                "derived_features": ["wmean"],
            }
        },
        "facial_tremor": {
            "group_1": {
                "raw_features": [f"fac_tremor_median_{i}" for i in [5, 12, 8, 48, 54, 28, 51, 66, 57]],
                "derived_features": ["mean"],
            }
        },
        "pause": {
            "group_1": {
                "raw_features": ["aco_pausetime", "aco_totaltime", "aco_numpauses", "aco_pausefrac"],
                "derived_features": ["mean"],
            }
        },
    }
    return feature_dict


def get_derived_fe_dict(group: str, subgroup: str):
    return _derived_fe_dict()[group][subgroup]


def get_fe_dict_structure() -> Dict[str, Dict[str, Dict[str, Sequence[str]]]]:
    fe_dict = _derived_fe_dict()
    info_dict = {x: len(fe_dict[x]) for x in fe_dict.keys()}
    return info_dict

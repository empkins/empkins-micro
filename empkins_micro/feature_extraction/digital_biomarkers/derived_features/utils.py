from itertools import product
from typing import Dict, Sequence


def _derived_fe_dict():
    feature_dict = {
        "facial": {
            "group_1": {
                "raw_features": [f"fac_LMK{i:02}disp" for i in range(0, 68)]
                + [f"fac_AU{i:02}int" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
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
            "group_1": {"raw_features": ["aco_jitter", "aco_shimmer", "aco_gne"],
                        "derived_features": ["mean", "wmean", "std", "wstd"]}
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
                "derived_features": ["mean", "wmean"],
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


def _subgroup_dict():
    subgroup_dict = {
        "fac": {
            "facial_landmarks": [f"LMK{i:02}disp" for i in range(0, 68)],
            "action_units":
                [f"AU{i:02}int" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
                + [f"AU{i:02}pres" for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]],
            "emotional_expressivity":
                [f"{em}int{intens}" for em, intens in
                    product(["hap", "sad", "sur", "fea", "ang", "dis", "con"], ["soft", "hard"])]
                + [f"{em}pres" for em in ["hap", "sad", "sur", "fea", "ang", "dis", "con"]],
            "overall_expressivity": [
                "comintsoft",
                "cominthard",
                "comlowintsoft",
                "comlowinthard",
                "comuppintsoft",
                "comuppinthard",
                "posintsoft",
                "posinthard",
                "negintsoft",
                "neginthard"
            ],
            "facial_asymmetry": [
                "asymmaskmouth",
                "asymmaskeye",
                "asymmaskeyebrow",
                "asymmaskcom"
            ],
            "pain_expressivity": [
                "paiintsoft",
                "paiinthard"
            ],
            "facial_tremor": [f"tremormedian{i}" for i in [5, 12, 8, 48, 54, 28, 51, 66, 57]]
        },
        "aco": {
            "fundamental_frequency": ["ff"],
            "formant_frequencies": [f"fm{i}" for i in range(1, 5)],
            "audio_intensity": ["int"],
            "hnr": ["hnr"],
            "gne": ["gne"],
            "jitter": ["jitter"],
            "shimmer": ["shimmer"],
            "pause": [
                "pausetime",
                "totaltime",
                "numpauses",
                "pausefrac"
            ],
            "mfcc": [f"mfcc{i}" for i in range(1, 13)],
            "voice_prevalence": ["voicepct"],
        },
        "mov": {
            "vocal_tremor": [
                "freqtremfreq",
                "freqtremindex",
                "freqtrempindex",
                "amptremfreq",
                "amptremindex",
                "amptrempindex"
            ],
            "head_movement": [
                "headvel",
                "hposepitch",
                "hposeyaw",
                "hposeroll",
                "hposedist"
            ],
            "eye_blink": [
                "eyeblink",
                "eyeblinkdur",
                "blinkear"
            ],
            "eye_gaze": [
                "lefteyex",
                "lefteyey",
                "lefteyez",
                "righteyex",
                "righteyey",
                "righteyez",
                "leyedisp",
                "reyedisp"
            ]
        }
    }

    return subgroup_dict


def get_subgroup(group, feature):
    subgroup_dict = _subgroup_dict()
    for subgroup in subgroup_dict[group]:
        if feature in subgroup_dict[group][subgroup]:
            return subgroup

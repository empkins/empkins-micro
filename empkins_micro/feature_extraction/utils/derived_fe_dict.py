
def _derived_fe_dict():
    feature_dict = {
        "facial": {
            "group_1": {
                "raw_features": [
                    # facial landmarks
                    "fac_LMK00disp",
                    "fac_LMK01disp",
                    "fac_LMK02disp",
                    "fac_LMK03disp",
                    "fac_LMK04disp",
                    "fac_LMK05disp",
                    "fac_LMK06disp",
                    "fac_LMK07disp",
                    "fac_LMK08disp",
                    "fac_LMK09disp",
                    "fac_LMK10disp",
                    "fac_LMK11disp",
                    "fac_LMK12disp",
                    "fac_LMK13disp",
                    "fac_LMK14disp",
                    "fac_LMK15disp",
                    "fac_LMK16disp",
                    "fac_LMK17disp",
                    "fac_LMK18disp",
                    "fac_LMK19disp",
                    "fac_LMK20disp",
                    "fac_LMK21disp",
                    "fac_LMK22disp",
                    "fac_LMK23disp",
                    "fac_LMK24disp",
                    "fac_LMK25disp",
                    "fac_LMK26disp",
                    "fac_LMK27disp",
                    "fac_LMK28disp",
                    "fac_LMK29disp",
                    "fac_LMK30disp",
                    "fac_LMK31disp",
                    "fac_LMK32disp",
                    "fac_LMK33disp",
                    "fac_LMK34disp",
                    "fac_LMK35disp",
                    "fac_LMK36disp",
                    "fac_LMK37disp",
                    "fac_LMK38disp",
                    "fac_LMK39disp",
                    "fac_LMK40disp",
                    "fac_LMK41disp",
                    "fac_LMK42disp",
                    "fac_LMK43disp",
                    "fac_LMK44disp",
                    "fac_LMK45disp",
                    "fac_LMK46disp",
                    "fac_LMK47disp",
                    "fac_LMK48disp",
                    "fac_LMK49disp",
                    "fac_LMK50disp",
                    "fac_LMK51disp",
                    "fac_LMK52disp",
                    "fac_LMK53disp",
                    "fac_LMK54disp",
                    "fac_LMK55disp",
                    "fac_LMK56disp",
                    "fac_LMK57disp",
                    "fac_LMK58disp",
                    "fac_LMK59disp",
                    "fac_LMK60disp",
                    "fac_LMK61disp",
                    "fac_LMK62disp",
                    "fac_LMK63disp",
                    "fac_LMK64disp",
                    "fac_LMK65disp",
                    "fac_LMK66disp",
                    "fac_LMK67disp",
                    # action units - mean, std
                    "fac_AU01int",
                    "fac_AU02int",
                    "fac_AU04int",
                    "fac_AU05int",
                    "fac_AU06int",
                    "fac_AU07int",
                    "fac_AU09int",
                    "fac_AU10int",
                    "fac_AU12int",
                    "fac_AU14int",
                    "fac_AU15int",
                    # "fac_AU16int", # not available in the raw dataset, but in documentation
                    "fac_AU17int",
                    "fac_AU20int",
                    "fac_AU23int",
                    "fac_AU25int",
                    "fac_AU26int",
                    # "fac_AU28int", # not available in the raw dataset, but fac_AU28pres
                    "fac_AU45int",
                    # emotional expressivity - mean, std
                    "fac_hapintsoft",
                    "fac_hapinthard",
                    "fac_sadintsoft",
                    "fac_sadinthard",
                    "fac_surintsoft",
                    "fac_surinthard",
                    "fac_feaintsoft",
                    "fac_feainthard",
                    "fac_angintsoft",
                    "fac_anginthard",
                    "fac_disintsoft",
                    "fac_disinthard",
                    "fac_conintsoft",
                    "fac_coninthard",
                    # facial asymmetry
                    "fac_asymmaskmouth",
                    "fac_asymmaskeyebrow",
                    "fac_asymmaskeye",
                    "fac_asymmaskcom"
                ],
                "derived_features": [
                    "mean",
                    "std"
                ]
            },
            "group_2": {
                "raw_features": [
                    # action units - pct
                    "fac_AU01pres",
                    "fac_AU02pres",
                    "fac_AU04pres",
                    "fac_AU05pres",
                    "fac_AU06pres",
                    "fac_AU07pres",
                    "fac_AU09pres",
                    "fac_AU10pres",
                    "fac_AU12pres",
                    "fac_AU14pres",
                    "fac_AU15pres",
                    # "fac_AU16pres", # not available in the raw dataset, but in documentation
                    "fac_AU17pres",
                    "fac_AU20pres",
                    "fac_AU23pres",
                    "fac_AU25pres",
                    "fac_AU26pres",
                    "fac_AU28pres",
                    "fac_AU45pres",
                    # emotional expressivity - pct
                    "fac_happres",
                    "fac_sadpres",
                    "fac_surpres",
                    "fac_feapres",
                    "fac_angpres",
                    "fac_dispres",
                    "fac_conpres"
                ],
                "derived_features": [
                    "pct"
                ]
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
                "derived_features": [
                    "mean",
                    "std",
                    "pct"
                ]
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
                "derived_features": [
                    "mean",
                    "std",
                    "range"
                ]
            },
            "group_2": {
                "raw_features": [
                    "aco_mfcc1",
                    "aco_mfcc2",
                    "aco_mfcc3",
                    "aco_mfcc4",
                    "aco_mfcc5",
                    "aco_mfcc6",
                    "aco_mfcc7",
                    "aco_mfcc8",
                    "aco_mfcc9",
                    "aco_mfcc10",
                    "aco_mfcc11",
                    "aco_mfcc12",
                ],
                "derived_features": [
                    "mean"
                ]
            }
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
                    "mov_reyedisp"
                ],
                "derived_features": [
                    "mean",
                    "std"
                ]
            },
            "group_2": {
                "raw_features": [
                    "mov_eyeblink"
                ],
                "derived_features": [
                    "count",
                    "dur_mean",
                    "dur_std"
                ]
            }
        },
        "eyeblink_ear": {
            "group_1": {
                "raw_features": [
                    "mov_blink_ear"
                ],
                "derived_features": [
                    "mean",
                    "std"
                ]
            }
        },
        "acoustic_seg": {
            "group_1": {
                "raw_features": [
                    "aco_jitter",
                    "aco_shimmer",
                    "aco_gne"
                ],
                "derived_features": [
                    "wmean",
                    "wstd"
                ]
            }
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
                    "aco_voicepct"
                ],
                "derived_features": [
                    "wmean"
                ]
            }
        },
        "facial_tremor": {
            "group_1": {
                "raw_features": [
                    "fac_tremor_median_5",
                    "fac_tremor_median_12",
                    "fac_tremor_median_8",
                    "fac_tremor_median_48",
                    "fac_tremor_median_54",
                    "fac_tremor_median_28",
                    "fac_tremor_median_51",
                    "fac_tremor_median_66",
                    "fac_tremor_median_57"
                ],
                "derived_features": [
                    "mean"
                ]
            }
        },
        "pause": {
            "group_1": {
                "raw_features": [
                    "aco_pausetime",
                    "aco_totaltime",
                    "aco_numpauses",
                    "aco_pausefrac"
                ],
                "derived_features": [
                    "mean"
                ]
            }
        }
    }
    return feature_dict


def get_derived_fe_dict(group: str, subgroup: str):
    return _derived_fe_dict()[group][subgroup]


def get_fe_dict_structure():
    fe_dict = _derived_fe_dict()
    info_dict = {x: len(fe_dict[x]) for x in fe_dict.keys()}
    return info_dict

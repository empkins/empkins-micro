
def _derived_fe_dict():
    feature_dict = {
        "facial": {
            "group_1": {
                "raw_features": [
                    "fac_asymmaskmouth",
                    "fac_asymmaskeyebrow",
                    "fac_asymmaskeye",
                    "fac_asymmaskcom",
                    "fac_AU01int",
                    "fac_AU02int",
                    "fac_AU04int",
                    "fac_AU05int",
                    "fac_AU06int",
                    "fac_AU07int",
                    "fac_AU09int",
                    "fac_AU010int",
                    "fac_AU012int",
                    "fac_AU014int",
                    "fac_AU015int",
                    "fac_AU016int",
                    "fac_AU017int",
                    "fac_AU020int",
                    "fac_AU023int",
                    "fac_AU025int",
                    "fac_AU026int",
                    "fac_AU028int",
                    "fac_AU045int",
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
                    "fac_cominthard",
                    "fac_comlowinthard",
                    "fac_comuppinthard",
                    "fac_posinthard",
                    "fac_neginthard",
                    "fac_paiinthard",
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
                ],
                "derived_features": [
                    "mean",
                    "std"
                ]
            },
            "group_2": {
                "raw_features": [
                    "fac_comintsoft",
                    "fac_comlowintsoft",
                    "fac_comuppintsoft",
                    "fac_posintsoft",
                    "fac_negintsoft",
                    "fac_paiintsoft"
                ],
                "derived_features": [
                    "mean",
                    "std",
                    "pct"
                ]
            },
            "group_3": {
                "raw_features": [
                    "fac_AU01_pres",
                    "fac_AU02_pres",
                    "fac_AU04_pres",
                    "fac_AU05_pres",
                    "fac_AU06_pres",
                    "fac_AU07_pres",
                    "fac_AU09_pres",
                    "fac_AU10_pres",
                    "fac_AU12_pres",
                    "fac_AU14_pres",
                    "fac_AU15_pres",
                    "fac_AU16_pres",
                    "fac_AU17_pres",
                    "fac_AU20_pres",
                    "fac_AU23_pres",
                    "fac_AU25_pres",
                    "fac_AU26_pres",
                    "fac_AU28_pres",
                    "fac_AU45_pres",
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
            }
        },
        "acoustic": {
            "group_1": {
                "raw_features": [
                    "aco_int",
                    "aco_ff",
                    "aco_hnr",
                    "aco_fm1",
                    "aco_fm2",
                    "aco_fm3",
                    "aco_fm4",
                ],
                "derived_features": [
                    "mean",
                    "std",
                    # "range"
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
                    "mov_headvel",
                    "mov_hposedist",
                    "mov_hposepitch",
                    "mov_hposeyaw",
                    "mov_hposeroll",
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
                    # TODO
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
                    "mean",
                    "std"
                ]
            }
        },
        "audio_seg": {
            "group_1": {
                "raw_features": [
                    "mov_freqtremfreq",
                    "mov_freqtremindex",
                    "mov_freqtrempindex",
                    "mov_amptremfreq",
                    "mov_amptremindex",
                    "mov_amptrempindex",
                    "aco_voicepct"
                ],
                "derived_features": [
                    "mean"
                ]
            },
            "group_2": {
                "raw_features": [
                    # TODO pause characteristics
                ],
                "derived_features": [
                    # TODO
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
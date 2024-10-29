from linear_regression import line_regression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut


RANDOM_SEED = 1


def _append(d, k, x):
    if k not in d:
        d[k] = [x]
    else:
        d[k].append(x)


def cross_validate(df, N=3, M=1, stratify=False):
    # model --> metrics (one array for each)
    model_results = {}

    for m in range(M):
        if stratify:
            # seed needs to increment to get different splits
            kf = StratifiedKFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED + m)
            splits = list(kf.split(df, df["subcat"]))
        else:
            kf = KFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED + m)
            splits = list(kf.split(df))

        for train_idxs, test_idxs in splits:
            df_train, df_test = df.iloc[train_idxs], df.iloc[test_idxs]

            # iterate over models
            def fit_mod(mod_strs, df_tr, df_te):
                for s in mod_strs:
                    if s not in model_results:
                        model_results[s] = {}
                    results = line_regression("predicted_complexity", s, df_tr, df_te)
                    for k, v in results.items():
                        _append(model_results[s], k, v)

            # ours
            model_strs = [
                "sqrt_seg_64points",
                "sqrt_num_classes",
                "sqrt_seg_64points + sqrt_num_classes",
                "sqrt_seg_64points_x_sqrt_num_classes",
                "sqrt_seg_64points_x_sqrt_num_classes + sqrt_seg_64points + sqrt_num_classes",
                "avg_region_similarity",
                "avg_rel_similarity",
                "avg_region_similarity + avg_rel_similarity",
                "sqrt_seg_64points + avg_region_similarity",
                "sqrt_seg_64points + avg_rel_similarity",
                "sqrt_num_classes + avg_region_similarity",
                "sqrt_num_classes + avg_rel_similarity",
                "sqrt_seg_64points + sqrt_num_classes + avg_region_similarity",
                "sqrt_seg_64points + sqrt_num_classes + avg_rel_similarity",
                "sqrt_seg_64points + sqrt_num_classes + avg_region_similarity + avg_rel_similarity",
                "seg_64points_norm",
                "seg_64points_norm + num_classes_norm",
                "num_classes_norm",
                "seg_64points_norm + num_classes_norm + avg_region_similarity",
                "seg_64points_norm + num_classes_norm + avg_rel_similarity",
                "seg_64points_norm + num_classes_norm + avg_region_similarity + avg_rel_similarity",
                "seg_64points_norm + num_classes_norm + avg_region_similarity_norm",
                "seg_64points_norm + num_classes_norm + avg_rel_similarity_norm",
                "seg_64points_norm + num_classes_norm + avg_region_similarity_norm + avg_rel_similarity_norm",
            ]

            fit_mod(model_strs, df_train, df_test)

            """
            # ablation
            model_strs = [
                "visc_symmetry",
                "sqrt_seg_64points + sqrt_num_classes + visc_symmetry",
            ]
            

            df_train_nona = df_train[~df_train["visc_symmetry"].isna()]
            df_test_nona = df_test[~df_test["visc_symmetry"].isna()]
            fit_mod(model_strs, df_train_nona, df_test_nona)
            """
    return model_results

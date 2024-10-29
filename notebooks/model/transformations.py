import numpy as np


def sqrt_xform(c):
    return np.sqrt(c)


def rank_xform(s):
    return s.rank(axis=0, method="average")


def get_sqrt_transforms(df):
    df["sqrt_seg_64points"] = sqrt_xform(df["num_seg_64points"])
    df["sqrt_num_classes"] = sqrt_xform(df["num_classes"])


def get_transforms(df):

    df["sqrt_seg_64points"] = sqrt_xform(df["num_seg_64points"])
    df["sqrt_num_classes"] = sqrt_xform(df["num_classes"])

    df["rank_seg_64points"] = rank_xform(df["num_seg_64points"])
    df["rank_num_classes"] = rank_xform(df["num_classes"])

    df["rank_seg_64points_x_num_classes"] = df["rank_seg_64points"] * df["num_classes"]
    df["rank_seg_64points_x_rank_num_classes"] = rank_xform(
        df["rank_seg_64points"] * df["rank_num_classes"]
    )
    df["sqrt_seg_64points_x_num_classes"] = df["sqrt_seg_64points"] * df["num_classes"]
    df["sqrt_seg_64points_x_sqrt_num_classes"] = (
        df["sqrt_seg_64points"] * df["sqrt_num_classes"]
    )

    # normalize
    df["seg_64points_norm"] = (
        df["num_seg_64points"] - df["num_seg_64points"].mean()
    ) / df["num_seg_64points"].std()
    df["num_classes_norm"] = (df["num_classes"] - df["num_classes"].mean()) / df[
        "num_classes"
    ].std()

    # normalize average similarity
    df["avg_region_similarity_norm"] = (
        df["avg_region_similarity"] - df["avg_region_similarity"].mean()
    ) / df["avg_region_similarity"].std()
    df["avg_rel_similarity_norm"] = (
        df["avg_rel_similarity"] - df["avg_rel_similarity"].mean()
    ) / df["avg_rel_similarity"].std()

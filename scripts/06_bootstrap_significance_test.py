

from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import PipelineModel


def build_spark():
    return (
        SparkSession.builder
        .appName("SUSY-Bootstrap-Test")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def bootstrap_auc(y_true, y_score, n_boot=200, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
       
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))

    scores = np.array(scores)
    mean_auc = float(scores.mean())
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return mean_auc, low, high, scores


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data/samples/susy_sample_parquet"

    lr_path = root / "models/logreg"
    gbt_path = root / "models/gbt"

    out_csv = root / "reports" / "bootstrap_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()

    df = spark.read.parquet(str(data_path)).select(col("label").cast("double"), col("features"))
    _, test = df.randomSplit([0.8, 0.2], seed=42)

    lr_model = PipelineModel.load(str(lr_path))
    gbt_model = PipelineModel.load(str(gbt_path))

    pred_lr = lr_model.transform(test)
    pred_gbt = gbt_model.transform(test)

  
    lr_pdf = (
        pred_lr
        .select(col("label"), vector_to_array(col("probability")).alias("prob_arr"))
        .select(col("label"), col("prob_arr")[1].alias("p1"))
        .toPandas()
    )

    gbt_pdf = (
        pred_gbt
        .select(col("label"), vector_to_array(col("probability")).alias("prob_arr"))
        .select(col("label"), col("prob_arr")[1].alias("p1"))
        .toPandas()
    )

    y = lr_pdf["label"].to_numpy()
    p_lr = lr_pdf["p1"].to_numpy()
    p_gbt = gbt_pdf["p1"].to_numpy()

    lr_mean, lr_low, lr_high, lr_scores = bootstrap_auc(y, p_lr, n_boot=200, seed=42)
    gbt_mean, gbt_low, gbt_high, gbt_scores = bootstrap_auc(y, p_gbt, n_boot=200, seed=43)

    m = min(len(lr_scores), len(gbt_scores))
    diffs = gbt_scores[:m] - lr_scores[:m]

    diff_mean = float(diffs.mean())
    diff_low = float(np.percentile(diffs, 2.5))
    diff_high = float(np.percentile(diffs, 97.5))

    p_value = float(np.mean(diffs <= 0))

    with open(out_csv, "w") as f:
        f.write("item,mean_auc,ci_lower,ci_upper\n")
        f.write(f"logreg,{lr_mean},{lr_low},{lr_high}\n")
        f.write(f"gbt,{gbt_mean},{gbt_low},{gbt_high}\n")
        f.write(f"gbt_minus_logreg,{diff_mean},{diff_low},{diff_high}\n")
        f.write(f"p_value,{p_value},,\n")

    print(" Saved:", out_csv)
    spark.stop()


if __name__ == "__main__":
    main()

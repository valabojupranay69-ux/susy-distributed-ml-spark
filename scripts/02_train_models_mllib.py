

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def build_spark(app_name: str = "SUSY-Train") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def eval_metrics(pred_df, label_col="label", pred_col="prediction", raw_col="rawPrediction", prob_col="probability"):
    acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="weightedPrecision")
    rec_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=pred_col, metricName="weightedRecall")

    metrics = {
        "accuracy": float(acc_eval.evaluate(pred_df)),
        "f1": float(f1_eval.evaluate(pred_df)),
        "precision": float(prec_eval.evaluate(pred_df)),
        "recall": float(rec_eval.evaluate(pred_df)),
    }

    try:
        auc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderROC")
        metrics["roc_auc"] = float(auc_eval.evaluate(pred_df))
    except Exception:
        metrics["roc_auc"] = None

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use the 100k sample parquet for quick runs.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / ("data/samples/susy_sample_parquet" if args.sample else "data/processed/susy_parquet")
    reports_dir = root / "reports"
    models_dir = root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("ERROR: data path not found:", data_path)
        return 1

    spark = build_spark()
    t0 = time.time()

    df = spark.read.parquet(str(data_path)).select(col("label").cast("double"), col("features"))
    df = df.repartition(200).cache()
    df.count()  

    train, test = df.randomSplit([args.train_ratio, 1 - args.train_ratio], seed=42)
    train = train.cache()
    test = test.cache()
    train.count(); test.count()

    models = {
        "logreg": LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.01),
        "rf": RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8),
        "gbt": GBTClassifier(labelCol="label", featuresCol="features", maxIter=50, maxDepth=5),
        "linsvc": LinearSVC(labelCol="label", featuresCol="features", maxIter=50, regParam=0.01),
    }

    rows = []
    for name, estimator in models.items():
        print(f"\n=== Training: {name} ===")
        start = time.time()

        pipe = Pipeline(stages=[estimator])
        fitted = pipe.fit(train)

        pred = fitted.transform(test)

        m = eval_metrics(pred)
        train_time = time.time() - start

        print("Metrics:", m)
        print(f"Train+Eval time (s): {train_time:.1f}")


        out_model = models_dir / name
        fitted.write().overwrite().save(str(out_model))

        rows.append((name, m["accuracy"], m["f1"], m["precision"], m["recall"], m["roc_auc"], train_time))


    out_csv = reports_dir / "model_metrics.csv"
    with open(out_csv, "w") as f:
        f.write("model,accuracy,f1,precision,recall,roc_auc,train_eval_seconds\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")

    elapsed = time.time() - t0
    print("\n Finished. Total time (min):", round(elapsed / 60, 2))
    print("Saved metrics to:", out_csv)
    print("Saved models to:", models_dir)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

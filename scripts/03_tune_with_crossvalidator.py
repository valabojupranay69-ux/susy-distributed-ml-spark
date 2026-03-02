

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def build_spark(app_name: str = "SUSY-CV-Tuning") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use 100k sample parquet.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = root / ("data/samples/susy_sample_parquet" if args.sample else "data/processed/susy_parquet")

    reports_dir = root / "reports"
    out_models = root / "models_tuned"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_models.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    t0 = time.time()

    df = spark.read.parquet(str(data_path)).select(col("label").cast("double"), col("features")).cache()
    df.count()

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train = train.cache()
    test = test.cache()
    train.count(); test.count()

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    results = []
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
    lr_pipe = Pipeline(stages=[lr])

    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .build()
    )

    lr_cv = CrossValidator(
        estimator=lr_pipe,
        estimatorParamMaps=lr_grid,
        evaluator=evaluator,
        numFolds=args.folds,
        parallelism=args.parallelism,
        seed=42
    )

    print("\n=== CrossValidator: LogisticRegression ===")
    start = time.time()
    lr_model = lr_cv.fit(train)
    lr_time = time.time() - start

    lr_best = lr_model.bestModel
    lr_pred = lr_best.transform(test)
    lr_auc = evaluator.evaluate(lr_pred)

    lr_best.write().overwrite().save(str(out_models / "logreg_cv"))
    results.append(("logreg_cv", len(lr_grid), args.folds, args.parallelism, float(lr_auc), lr_time))


    gbt = GBTClassifier(labelCol="label", featuresCol="features")
    gbt_pipe = Pipeline(stages=[gbt])

    gbt_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 5])
        .addGrid(gbt.maxIter, [30, 60])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )

    gbt_cv = CrossValidator(
        estimator=gbt_pipe,
        estimatorParamMaps=gbt_grid,
        evaluator=evaluator,
        numFolds=args.folds,
        parallelism=args.parallelism,
        seed=42
    )

    print("\n=== CrossValidator: GBTClassifier ===")
    start = time.time()
    gbt_model = gbt_cv.fit(train)
    gbt_time = time.time() - start

    gbt_best = gbt_model.bestModel
    gbt_pred = gbt_best.transform(test)
    gbt_auc = evaluator.evaluate(gbt_pred)

    gbt_best.write().overwrite().save(str(out_models / "gbt_cv"))
    results.append(("gbt_cv", len(gbt_grid), args.folds, args.parallelism, float(gbt_auc), gbt_time))

    out_csv = reports_dir / "tuning_results.csv"
    with open(out_csv, "w") as f:
        f.write("model,grid_size,folds,parallelism,test_roc_auc,seconds\n")
        for r in results:
            f.write(",".join(map(str, r)) + "\n")

    elapsed = time.time() - t0
    print("\n Done. Total time (min):", round(elapsed / 60, 2))
    print("Saved:", out_csv)

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

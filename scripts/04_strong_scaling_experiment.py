

from pathlib import Path
import time
import sys

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline


def build_spark(partitions: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(f"SUSY-Scaling-{partitions}")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def run_experiment(partitions: int):
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data/samples/susy_sample_parquet"

    spark = build_spark(partitions)

    df = spark.read.parquet(str(data_path)).cache()
    df.count()

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
    pipe = Pipeline(stages=[lr])

    start = time.time()
    model = pipe.fit(train)
    elapsed = time.time() - start

    spark.stop()

    return elapsed


def main():
    root = Path(__file__).resolve().parents[1]
    out_csv = root / "reports/strong_scaling_results.csv"

    partitions_list = [50, 100, 200, 400]

    results = []

    for p in partitions_list:
        print(f"\nRunning with shuffle partitions = {p}")
        t = run_experiment(p)
        print(f"Time: {t:.2f} seconds")
        results.append((p, t))

    with open(out_csv, "w") as f:
        f.write("shuffle_partitions,train_time_seconds\n")
        for r in results:
            f.write(f"{r[0]},{r[1]}\n")

    print("\nSaved results to:", out_csv)


if __name__ == "__main__":
    main()



from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline


def build_spark(partitions: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(f"SUSY-Weak-Scaling-{partitions}")
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.adaptive.enabled", "true")
    
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


def train_lr(df):

    df = df.select("label", "features")

  
    df = df.repartition(200).cache()
    df.count()  

    train, _ = df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.01)
    pipe = Pipeline(stages=[lr])

    start = time.time()
    _ = pipe.fit(train)
    elapsed = time.time() - start

    df.unpersist()
    return elapsed


def main():
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "data/samples/susy_sample_parquet"
    full_path = root / "data/processed/susy_parquet"
    out_csv = root / "reports/weak_scaling_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    partitions = 200  

    spark = build_spark(partitions)

    results = []

  
    df_100k = spark.read.parquet(str(sample_path))
    t_100k = train_lr(df_100k)
    results.append(("100k_sample", 100_000, partitions, t_100k))

 
    df_full = spark.read.parquet(str(full_path))
    df_1m = df_full.sample(withReplacement=False, fraction=0.20, seed=42)
    t_1m = train_lr(df_1m)
    results.append(("~1m_sample_fraction0.20", "~1,000,000", partitions, t_1m))

   
    df_full2 = spark.read.parquet(str(full_path)).select("label", "features").repartition(200)

    train_full, _ = df_full2.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50, regParam=0.01)
    pipe = Pipeline(stages=[lr])

    start = time.time()
    _ = pipe.fit(train_full)
    t_full = time.time() - start
    results.append(("5m_full_nocache", 5_000_000, partitions, t_full))

    with open(out_csv, "w") as f:
        f.write("dataset,rows,shuffle_partitions,train_time_seconds\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    print(" Saved:", out_csv)
    spark.stop()


if __name__ == "__main__":
    main()

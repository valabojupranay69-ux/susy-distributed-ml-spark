

from pathlib import Path
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler


def build_spark(app_name: str = "SUSY-Ingest") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "SUSY.csv.gz"
    out_parquet = root / "data" / "processed" / "susy_parquet"
    out_sample = root / "data" / "samples" / "susy_sample_parquet"

    if not raw_path.exists():
        print(f"ERROR: Missing input file: {raw_path}")
        return 1

    out_parquet.mkdir(parents=True, exist_ok=True)
    out_sample.mkdir(parents=True, exist_ok=True)

    spark = build_spark()
    t0 = time.time()


    tmp_cols = [f"c{i}" for i in range(19)]

    df = (
        spark.read
        .option("inferSchema", "true")
        .option("header", "false")
        .csv(str(raw_path))
        .toDF(*tmp_cols)
    )


    for c in tmp_cols:
        df = df.withColumn(c, col(c).cast(DoubleType()))


    label_col = "label"
    feature_cols = [f"f{i}" for i in range(1, 19)]  

    rename_exprs = [col("c0").alias(label_col)]
    rename_exprs += [col(f"c{i}").alias(feature_cols[i-1]) for i in range(1, 19)]
    df = df.select(*rename_exprs)

  
    print("Row count (may take time first run)...")
    n = df.count()
    print("Rows:", n)
    print("Columns:", len(df.columns), df.columns)

  
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df).select(label_col, "features")

  
    df_vec.write.mode("overwrite").parquet(str(out_parquet))
    print("Wrote Parquet to:", out_parquet)

   
    df_sample = df_vec.limit(100_000)
    df_sample.write.mode("overwrite").parquet(str(out_sample))
    print("Wrote sample Parquet to:", out_sample)

    elapsed = time.time() - t0
    print(f"Done Ingestion time: {elapsed/60:.2f} minutes")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

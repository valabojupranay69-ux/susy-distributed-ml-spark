

from pathlib import Path
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.functions import vector_to_array

try:
    from pyspark.ml.functions import array_to_vector
    HAS_ARRAY_TO_VECTOR = True
except Exception:
    HAS_ARRAY_TO_VECTOR = False
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors
    from pyspark.sql.types import ArrayType, DoubleType
    from pyspark.sql.types import VectorUDT  


def build_spark():
    return (
        SparkSession.builder
        .appName("SUSY-Stability-Test")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]

   
    data_path = root / "data/samples/susy_sample_parquet"

    model_path = root / "models/gbt"

    out_csv = root / "reports" / "stability_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark()

    df = spark.read.parquet(str(data_path)).select(col("label").cast("double"), col("features"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    model = PipelineModel.load(str(model_path))

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )


    pred_base = model.transform(test).select("label", "features", "prediction", "rawPrediction", "probability")
    auc_base = float(evaluator.evaluate(pred_base))


    base = pred_base.withColumn("f_arr", vector_to_array(col("features")))


    sigmas = [0.001, 0.01, 0.05]

    rows = []
    for sigma in sigmas:
        t0 = time.time()

    
        noisy_arr = expr(f"transform(f_arr, x -> x + (randn() * {sigma}))")
        df_noisy = base.withColumn("f_noisy_arr", noisy_arr)

        if HAS_ARRAY_TO_VECTOR:
            df_noisy = df_noisy.withColumn("features_noisy", array_to_vector(col("f_noisy_arr")))
        else:
          
            to_vec = udf(lambda xs: Vectors.dense(xs), VectorUDT())
            df_noisy = df_noisy.withColumn("features_noisy", to_vec(col("f_noisy_arr")))

       
        test_noisy = df_noisy.select(col("label"), col("features_noisy").alias("features"), col("prediction").alias("pred_base"))

        pred_noisy = model.transform(test_noisy).select("label", col("prediction").alias("pred_noisy"), "rawPrediction", "probability", "pred_base")

        auc_noisy = float(evaluator.evaluate(pred_noisy))

      
        flips = pred_noisy.where(col("pred_noisy") != col("pred_base")).count()
        total = pred_noisy.count()
        flip_rate = float(flips / total) if total else 0.0

        elapsed = time.time() - t0
        rows.append((sigma, auc_base, auc_noisy, auc_noisy - auc_base, flip_rate, elapsed))

        print(f"sigma={sigma} | auc_base={auc_base:.6f} auc_noisy={auc_noisy:.6f} "
              f"delta={auc_noisy-auc_base:+.6f} flip_rate={flip_rate:.4f} time={elapsed:.1f}s")

    with open(out_csv, "w") as f:
        f.write("sigma,auc_base,auc_noisy,auc_delta,flip_rate,seconds\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(" Saved:", out_csv)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

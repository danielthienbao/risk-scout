"""Compute supply-chain risk scores from cleaned data using Spark.

Falls back to pandas when local Spark runtime is unavailable.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand, when, greatest, least


def _resolve_input_csv(processed_dir: Path) -> Path:
    spark_output_dir = processed_dir / 'cleaned_supply_chain_logs.csv'
    if spark_output_dir.is_dir():
        part_files = sorted(spark_output_dir.glob('part-*.csv'))
        if part_files:
            return part_files[0]
    return spark_output_dir


def _resolve_named_csv(processed_dir: Path, name: str) -> Path:
    candidate = processed_dir / name
    if candidate.is_dir():
        part_files = sorted(candidate.glob('part-*.csv'))
        if part_files:
            return part_files[0]
    return candidate


def _compute_with_pandas(processed_dir: Path) -> None:
    input_csv = _resolve_input_csv(processed_dir)
    df = pd.read_csv(input_csv)
    market_input_csv = _resolve_named_csv(processed_dir, 'cleaned_market_data.csv')
    if market_input_csv.exists():
        market_df = pd.read_csv(market_input_csv)[['entity_id', 'region']]
        df = df.merge(market_df, on='entity_id', how='left')
        df['region'] = df['region'].fillna('unknown')

    # External factor is sampled uniformly from [0, 100] for each row.
    rng = np.random.default_rng(seed=42)
    external_risk_factor = rng.uniform(0, 100, size=len(df))

    df['risk_score'] = (
        (df['delay_rate'] * 0.35)
        + (df['weather_risk'] * 0.25)
        + ((100 - df['supplier_reliability']) * 0.25)
        + (external_risk_factor * 0.15)
    ).clip(lower=0, upper=100)

    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[-0.001, 30, 70, 100],
        labels=['Low', 'Medium', 'High'],
    )

    df.to_csv(processed_dir / 'risk_scores.csv', index=False)
    print('Spark unavailable; wrote risk scores with pandas fallback.')


def main():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / 'data' / 'processed'

    try:
        spark = (
            SparkSession.builder
            .appName('RiskScoreComputation')
            .master('local[*]')
            .getOrCreate()
        )

        supply_df = spark.read.option('header', True).option('inferSchema', True).csv(
            str(processed_dir / 'cleaned_supply_chain_logs.csv')
        )
        market_df = spark.read.option('header', True).option('inferSchema', True).csv(
            str(processed_dir / 'cleaned_market_data.csv')
        ).select('entity_id', 'region')

        supply_df = supply_df.join(market_df, on='entity_id', how='left')

        external_risk_factor = rand(seed=42) * lit(100.0)

        scored_df = supply_df.withColumn(
            'risk_score_raw',
            (col('delay_rate') * lit(0.35))
            + (col('weather_risk') * lit(0.25))
            + ((lit(100.0) - col('supplier_reliability')) * lit(0.25))
            + (external_risk_factor * lit(0.15)),
        )

        scored_df = scored_df.withColumn(
            'risk_score',
            least(lit(100.0), greatest(lit(0.0), col('risk_score_raw'))),
        ).drop('risk_score_raw')

        scored_df = scored_df.withColumn(
            'risk_level',
            when(col('risk_score') <= 30, 'Low')
            .when((col('risk_score') > 30) & (col('risk_score') <= 70), 'Medium')
            .otherwise('High'),
        )

        scored_df.coalesce(1).write.mode('overwrite').option('header', True).csv(
            str(processed_dir / 'risk_scores.csv')
        )

        print('Risk scores written to data/processed/risk_scores.csv with Spark.')
        spark.stop()
    except Exception as exc:
        print(f'Spark risk scoring failed ({exc}). Falling back to pandas.')
        _compute_with_pandas(processed_dir)


if __name__ == '__main__':
    main()

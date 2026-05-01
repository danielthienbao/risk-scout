"""Ingest and clean raw market and supply-chain data with Spark.

Falls back to pandas when local Spark runtime is unavailable.
"""

from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


def fill_missing_values(df, numeric_cols, string_cols):
    """Fill missing numeric and string values with sensible defaults."""
    for c in numeric_cols:
        df = df.withColumn(c, when(col(c).isNull(), 0.0).otherwise(col(c)))
    for c in string_cols:
        df = df.withColumn(c, when(col(c).isNull(), "unknown").otherwise(col(c)))
    return df


def _ingest_with_pandas(raw_dir: Path, processed_dir: Path) -> None:
    """Fallback ingestion path for environments where Spark cannot initialize."""
    market_df = pd.read_csv(raw_dir / 'market_data.csv')
    supply_df = pd.read_csv(raw_dir / 'supply_chain_logs.csv')

    market_numeric = ['volatility', 'liquidity_risk', 'market_sentiment']
    market_string = ['entity_id', 'entity_type', 'region', 'timestamp']
    supply_numeric = ['delay_rate', 'weather_risk', 'supplier_reliability']
    supply_string = ['entity_id', 'route', 'origin', 'destination', 'timestamp']

    market_df[market_numeric] = market_df[market_numeric].fillna(0.0)
    market_df[market_string] = market_df[market_string].fillna('unknown')
    supply_df[supply_numeric] = supply_df[supply_numeric].fillna(0.0)
    supply_df[supply_string] = supply_df[supply_string].fillna('unknown')

    market_df.to_csv(processed_dir / 'cleaned_market_data.csv', index=False)
    supply_df.to_csv(processed_dir / 'cleaned_supply_chain_logs.csv', index=False)
    print('Spark unavailable; wrote cleaned datasets with pandas fallback.')


def main():
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        spark = (
            SparkSession.builder
            .appName('RiskIntelligenceIngestion')
            .master('local[*]')
            .getOrCreate()
        )

        market_df = spark.read.option('header', True).option('inferSchema', True).csv(str(raw_dir / 'market_data.csv'))
        supply_df = spark.read.option('header', True).option('inferSchema', True).csv(str(raw_dir / 'supply_chain_logs.csv'))

        cleaned_market_df = fill_missing_values(
            market_df,
            numeric_cols=['volatility', 'liquidity_risk', 'market_sentiment'],
            string_cols=['entity_id', 'entity_type', 'region', 'timestamp'],
        )

        cleaned_supply_df = fill_missing_values(
            supply_df,
            numeric_cols=['delay_rate', 'weather_risk', 'supplier_reliability'],
            string_cols=['entity_id', 'route', 'origin', 'destination', 'timestamp'],
        )

        cleaned_market_df.coalesce(1).write.mode('overwrite').option('header', True).csv(
            str(processed_dir / 'cleaned_market_data.csv')
        )
        cleaned_supply_df.coalesce(1).write.mode('overwrite').option('header', True).csv(
            str(processed_dir / 'cleaned_supply_chain_logs.csv')
        )

        print('Cleaned datasets written to data/processed with Spark.')
        spark.stop()
    except Exception as exc:
        print(f'Spark ingestion failed ({exc}). Falling back to pandas.')
        _ingest_with_pandas(raw_dir, processed_dir)


if __name__ == '__main__':
    main()

"""
Parquet Loader for Indian Supreme Court Judgments

Loads parquet files into pandas DataFrames and extracts required fields.

Usage:
    from src.ingest.parquet_loader import load_parquet_file, load_all_parquets
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


# Required fields to extract from parquet files
REQUIRED_FIELDS = [
    'judgment_text',
    'case_number',
    'judges',
    'date_of_judgment'
]


def load_parquet_file(filepath: str) -> pd.DataFrame:
    """
    Load a single parquet file and extract required fields.
    
    Args:
        filepath: Path to the parquet file
        
    Returns:
        pd.DataFrame: DataFrame with required fields
        
    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If required fields are missing
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    print(f"📂 Loading parquet: {filepath.name}")
    
    try:
        # Load parquet file
        df = pd.read_parquet(filepath)
        print(f"   Total rows: {len(df)}")
        
        # Check for required fields
        missing_fields = [field for field in REQUIRED_FIELDS if field not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Extract only required fields
        df = df[REQUIRED_FIELDS].copy()
        
        # Drop rows with empty judgment_text (critical field)
        initial_count = len(df)
        df = df.dropna(subset=['judgment_text'])
        df = df[df['judgment_text'].str.strip() != '']
        dropped = initial_count - len(df)
        
        if dropped > 0:
            print(f"   Dropped {dropped} rows with empty judgment_text")
        
        print(f"   Valid rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading {filepath.name}: {e}")
        raise


def load_all_parquets(directory: str = "data/parquet") -> pd.DataFrame:
    """
    Load all parquet files from a directory and combine them.
    
    Args:
        directory: Directory containing parquet files
        
    Returns:
        pd.DataFrame: Combined DataFrame from all parquet files
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all parquet files
    parquet_files = list(directory.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    
    print(f"\n🔍 Found {len(parquet_files)} parquet file(s) in {directory}")
    print("-" * 60)
    
    dataframes = []
    total_rows = 0
    
    for parquet_file in sorted(parquet_files):
        try:
            df = load_parquet_file(parquet_file)
            dataframes.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"⚠️  Skipping {parquet_file.name} due to error: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No valid parquet files could be loaded")
    
    # Combine all dataframes
    print("-" * 60)
    print(f"📊 Combining {len(dataframes)} dataframe(s)...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"✅ Total judgments loaded: {len(combined_df)}")
    print(f"   Fields: {list(combined_df.columns)}")
    
    return combined_df


def get_judgment_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get statistics about the loaded judgments.
    
    Args:
        df: DataFrame containing judgments
        
    Returns:
        Dict with statistics
    """
    stats = {
        'total_judgments': len(df),
        'fields': list(df.columns),
        'avg_text_length': df['judgment_text'].str.len().mean() if 'judgment_text' in df.columns else 0,
        'max_text_length': df['judgment_text'].str.len().max() if 'judgment_text' in df.columns else 0,
        'min_text_length': df['judgment_text'].str.len().min() if 'judgment_text' in df.columns else 0,
        'null_counts': df.isnull().sum().to_dict()
    }
    
    return stats


def print_stats(df: pd.DataFrame):
    """Print statistics about loaded judgments."""
    stats = get_judgment_stats(df)
    
    print("\n📈 Dataset Statistics:")
    print(f"   Total judgments: {stats['total_judgments']:,}")
    print(f"   Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"   Max text length: {stats['max_text_length']:,} chars")
    print(f"   Min text length: {stats['min_text_length']:,} chars")
    
    if any(stats['null_counts'].values()):
        print(f"\n   Null values per field:")
        for field, count in stats['null_counts'].items():
            if count > 0:
                print(f"     - {field}: {count}")


if __name__ == "__main__":
    """Test the parquet loader."""
    import sys
    
    try:
        # Load all parquet files
        df = load_all_parquets("data/parquet")
        
        # Print statistics
        print_stats(df)
        
        # Show sample judgment
        if len(df) > 0:
            print("\n📄 Sample Judgment:")
            sample = df.iloc[0]
            print(f"   Case Number: {sample['case_number']}")
            print(f"   Judges: {sample['judges']}")
            print(f"   Date: {sample['date_of_judgment']}")
            print(f"   Text Preview: {str(sample['judgment_text'])[:200]}...")
        
        print("\n✅ Parquet loader test successful!")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

import pandas as pd
import ta

def load_data(filepath: str) -> pd.DataFrame:
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Use 'datetime' column instead of 'Date'
    if 'datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        raise KeyError("No datetime or timestamp column found in CSV")
    
    # Set index
    df.set_index('Date', inplace=True)
    
    # Ensure column names are capitalized for consistency
    df.rename(columns=lambda x: x.capitalize(), inplace=True)
    
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Add simple moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Add RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    return df

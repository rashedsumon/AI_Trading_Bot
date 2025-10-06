import pandas as pd
import ta

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Add simple moving average
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    return df

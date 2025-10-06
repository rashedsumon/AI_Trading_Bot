import streamlit as st
import pandas as pd
from bot.utils import load_data, add_indicators
from bot.ai_model import train_model
from bot.trading_bot import TradingBot

st.title("AI Cryptocurrency Trading Bot")

# Load data
data = load_data("data/Binance_Data.csv")
data = add_indicators(data)
st.subheader("Data Preview")
st.dataframe(data.tail())

# Train AI model
if st.button("Train AI Model"):
    model = train_model(data)
    st.success("Model trained and saved!")

# Initialize bot
initial_balance = st.number_input("Initial Balance ($)", min_value=100.0, value=1000.0)
bot = TradingBot(balance=initial_balance)

if st.button("Run Simulation"):
    pnl_list = []
    for idx, row in data.iterrows():
        features = [row['SMA_20'], row['SMA_50'], row['RSI']]
        signal = bot.predict_signal(features)
        pnl = bot.execute_trade(signal)
        pnl_list.append(pnl)
    
    st.subheader("Simulation Results")
    st.write(f"Final Balance: ${bot.balance:.2f}")
    st.write(f"Total Loss: ${bot.total_loss:.2f}")
    st.line_chart(pd.Series(pnl_list).cumsum(), use_container_width=True)

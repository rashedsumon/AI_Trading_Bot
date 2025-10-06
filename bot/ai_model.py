from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(df):
    df = df.dropna()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # next candle up or down

    features = ['SMA_20', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.2f}")

    joblib.dump(model, "bot/trading_model.pkl")
    return model

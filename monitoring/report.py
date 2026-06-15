import json
import pandas as pd

LOG_FILE = "monitoring/predictions.json"

def generate_report():
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\n📊 Prediction Summary")
    print(f"Total predictions:     {len(df)}")
    print(f"Cat predictions:       {(df['prediction'] == 'cat').sum()}")
    print(f"Dog predictions:       {(df['prediction'] == 'dog').sum()}")
    print(f"Avg confidence:        {df['confidence'].mean():.2f}%")
    print(f"Low confidence (<60%): {(df['confidence'] < 60).sum()} predictions")
    print(f"First prediction:      {df['timestamp'].min()}")
    print(f"Last prediction:       {df['timestamp'].max()}")

    # Drift detection — compare avg confidence over time
    if len(df) >= 10:
        mid = len(df) // 2
        early_conf = df.iloc[:mid]["confidence"].mean()
        late_conf  = df.iloc[mid:]["confidence"].mean()
        drift      = early_conf - late_conf

        print(f"\n📉 Confidence Drift")
        print(f"Early avg confidence:  {early_conf:.2f}%")
        print(f"Recent avg confidence: {late_conf:.2f}%")
        print(f"Drift:                 {drift:+.2f}%")

        if drift > 10:
            print("⚠️  WARNING: Confidence dropping — consider retraining!")
        else:
            print("✅ Model looks stable")
    else:
        print(f"\n⚠️  Need at least 10 predictions to detect drift (have {len(df)})")

if __name__ == "__main__":
    generate_report()
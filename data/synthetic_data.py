import numpy as np
import pandas as pd
import os

def generate_synthetic_data(num_days=30, output_path="synthetic_bess_data.csv"):
    """
    Generate synthetic time-series data for BESS RL Environment.
    Features:
    - price: Electricity price ($/MWh), higher during peak hours (e.g. 17:00-21:00)
    - load: Building/Grid load (MW), peak daytime.
    - frequency: Grid frequency (Hz), centered around 50 or 60. We use a standardized deviation -1.0 to 1.0.
    """
    total_hours = num_days * 24
    
    # Time
    hours_of_day = np.arange(total_hours) % 24
    
    # Price ($/MWh)
    # Base price around $40. Peak price (17-21) goes up to $150.
    base_price = np.random.normal(40, 5, total_hours)
    peak_multiplier = np.where((hours_of_day >= 17) & (hours_of_day <= 21), 3.0, 1.0)
    price = base_price * peak_multiplier + np.random.normal(0, 10, total_hours)
    price = np.clip(price, 10, None) # Min price $10
    
    # Load (MW)
    # Base load around 10MW, peaks during day 9-18
    base_load = np.random.normal(10, 2, total_hours)
    load_multiplier = np.where((hours_of_day >= 9) & (hours_of_day <= 18), 1.5, 1.0)
    load = base_load * load_multiplier + np.random.normal(0, 1.5, total_hours)
    load = np.clip(load, 2, None)
    
    # Frequency Deviation (Hz)
    # Ideally 0. Deviates occasionally.
    # Simple autoregressive model for frequency to simulate persistence
    freq_dev = np.zeros(total_hours)
    freq_dev[0] = np.random.normal(0, 0.05)
    for i in range(1, total_hours):
        freq_dev[i] = 0.7 * freq_dev[i-1] + np.random.normal(0, 0.02)
    
    freq_dev = np.clip(freq_dev, -0.5, 0.5)
    
    df = pd.DataFrame({
        "hour_of_day": hours_of_day,
        "price": price,
        "load": load,
        "frequency_deviation": freq_dev
    })
    
    # Ensure directory exists if saving
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic data to {output_path}")
        
    return df

if __name__ == "__main__":
    generate_synthetic_data(num_days=365, output_path=os.path.join(os.path.dirname(__file__), "synthetic_bess_data.csv"))

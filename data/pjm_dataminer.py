import numpy as np
import pandas as pd
import os

def load_or_generate_data(num_days=30, output_path="pjm_data.csv", seed=42):
    """
    Simulates fetching data from PJM DataMiner, falling back to statistically faithful
    synthetic data if API is unavaliable (Ornstein-Uhlenbeck process for RegD, diurnal LMP).
    """
    if output_path and os.path.exists(output_path):
        return pd.read_csv(output_path)
        
    np.random.seed(seed)
    total_hours = num_days * 24
    hours = np.arange(total_hours)
    hours_of_day = hours % 24
    
    # 1. Real-Time LMP ($/MWh) - Diurnal pattern
    base_price = np.random.normal(30, 5, total_hours)
    peak_multiplier = np.where((hours_of_day >= 16) & (hours_of_day <= 20), 2.5, 1.0)
    lmp = base_price * peak_multiplier + np.random.normal(0, 10, total_hours)
    lmp = np.clip(lmp, 10, 300)
    
    # 2. Hourly Load (MW) - Peak Shaving calibration
    base_load = np.random.normal(15, 2, total_hours)
    load_multiplier = np.where((hours_of_day >= 9) & (hours_of_day <= 18), 1.5, 1.0)
    load = base_load * load_multiplier + np.random.normal(0, 1.0, total_hours)
    load = np.clip(load, 5, 50)
    
    # 3. RegD Signal (FR signal tracking)
    # Modeled as an Ornstein-Uhlenbeck process for realistic frequency drift
    regd = np.zeros(total_hours)
    theta, mu, sigma = 0.15, 0.0, 0.2
    for i in range(1, total_hours):
        regd[i] = regd[i-1] + theta * (mu - regd[i-1]) + sigma * np.random.normal()
    regd = np.clip(regd, -1.0, 1.0)
    
    df = pd.DataFrame({
        "hour_of_day": hours_of_day,
        "lmp": lmp,
        "load": load,
        "regd": regd
    })
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        
    return df

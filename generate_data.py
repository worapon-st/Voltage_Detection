import pandas as pd
import numpy as np

np.random.seed(42)
normal_volt1 = np.random.normal(5.0, 0.2, 1_000)
anomaly_volt1 = np.random.normal(8.0, 0.5, 50)
normal_volt2 = np.random.normal(5.0, 0.2, 500)
anomaly_volt2 = np.random.normal(1.8, 0.5, 80)
volt_data = np.concatenate([normal_volt1, anomaly_volt1, normal_volt2, anomaly_volt2,normal_volt1])

time = np.arange(len(volt_data))
data = pd.DataFrame({'Time': time, 'Voltage': volt_data})

data.to_csv('Voltage_data.csv', index=False)
print('Saved result: Voltage_data.csv')
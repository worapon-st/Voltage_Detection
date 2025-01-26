import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.ensemble import IsolationForest


data = pd.read_csv('D:/PROJECT/Voltage_data.csv')

#Create Anomaly Detection by using IsolationForest
model = IsolationForest(contamination=0.05, random_state=42)
data['Anomaly'] = model.fit_predict(data[['Voltage']])

#separate normal and anomaly dataset
normal = data[data['Anomaly'] == 1]
anomaly = data[data['Anomaly'] == -1]

#find average high and low anomaly
anomaly_than = anomaly[anomaly['Voltage'] > 5.0]
anomaly_less = anomaly[anomaly['Voltage'] < 5.0]

mean_anomaly_than = round(anomaly_than['Voltage'].mean(), 2)
mean_anomaly_less = round(anomaly_less['Voltage'].mean(), 2)

print(f'Anomaly: \nHigh: {mean_anomaly_than} V \nLow:  {mean_anomaly_less} V')

#Visualize Anomaly Voltage
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['Voltage'], label='Voltage Data')
plt.plot(anomaly['Time'], anomaly['Voltage'], 'r.', label='Anomaly')
plt.title('Voltage Data With Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.legend()
savefig('Voltage_Anomaly')
plt.show()

#save data
data.to_csv('Volt_anomaly_detect.csv', index=False)
print('saved result: Volt_anomaly_detect.csv')

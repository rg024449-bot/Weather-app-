# Weather pattern analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("Project/weather_data.csv")  # file ko load karadiya

#data cleaning
# Null ya missing data points ko replace karooo
df['Temperature'] = df['Temperature'].replace(np.nan, df['Temperature'].mean())
df['Humidity'] = df['Humidity'].replace(np.nan, df['Humidity'].mean())
df['WindSpeed'] = df['WindSpeed'].replace(np.nan, df['WindSpeed'].mean())
df['Rainfall'] = df['Rainfall'].replace(np.nan, df['Rainfall'].mean())

# date time month theek karo
df['Date'] = pd.to_datetime(df['Date'] )  # date time format me convert kiya
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

#Average values show karo
temp_mean = np.mean(df['Temperature'])
temp_std = np.std(df['Temperature'])
humid_mean = np.mean(df['Humidity'])

print(f"\nAverage Temperature: {temp_mean:.2f} °C")
print(f"Temperature Std Dev: {temp_std:.2f}")
print(f"Average Humidity: {humid_mean:.2f}%")

monthly_avg = df.groupby('Month')[['Temperature', 'Humidity', 'Rainfall']].mean()
print("\nMonthly Average Weather Data:")
print(monthly_avg)

#matplotlib visualisation
plt.figure(figsize=(10,5))
plt.plot(monthly_avg.index, monthly_avg['Temperature'], marker='o', color='red', label='Temperature (°C)')
plt.plot(monthly_avg.index, monthly_avg['Humidity'], marker='s', color='blue', label='Humidity (%)')
plt.title('Monthly Average Temperature & Humidity')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

#heatmap seaborn
plt.figure(figsize=(6,4))
sns.heatmap(df[['Temperature','Humidity','WindSpeed','Rainfall']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Weather Feature Correlation')
plt.show()

# Pairplot seaborn
sns.pairplot(df[['Temperature','Humidity','WindSpeed','Rainfall']])
plt.suptitle('Weather Feature Relationships', y=1.02)
plt.show()

#summary
print("\n--- Insights Summary ---")
print(f"Highest average temperature month: {monthly_avg['Temperature'].idxmax()}")
print(f"Highest average rainfall month: {monthly_avg['Rainfall'].idxmax()}")
print(f"Lowest average temperature month: {monthly_avg['Temperature'].idxmin()}")
print("Correlation between temperature and humidity:")
print(df[['Temperature', 'Humidity']].corr())

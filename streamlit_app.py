import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# -----------------------------
#  Your model functions
# -----------------------------
def yearly_sine(t, A, phi, C):
    N = len(t)
    return A * np.sin(2 * np.pi * t / N + phi) + C

def daily_cosines(t, A1, phi1, A2, phi2, A3, phi3, C):
    w = 2 * np.pi / 24
    return (A1*np.cos(w*t + phi1) +
            A2*np.cos(2*w*t + phi2) +
            A3*np.cos(3*w*t + phi3) +
            C)

# -----------------------------
#  Load your fitted polynomials
# -----------------------------
Ay_poly   = np.array([ 3.65903488e-01, -7.37313256e+02])
phiy_poly = np.array([-3.75011208e-02,  7.87505260e+01])
Cy_poly   = np.array([ 2.25244689e+00, -4.52737343e+03])

daily_polys = [
    np.array([ 3.01673536e-01, -6.05745088e+02]),  # A1
    np.array([-5.31069944e-03,  1.49179803e+01]),  # phi1
    np.array([ 1.37938724e-01, -2.77836252e+02]),  # A2
    np.array([-1.51019030e-02,  3.14952481e+01]),  # phi2
    np.array([ 7.35584660e-02, -1.50800371e+02]),  # A3
    np.array([ 7.11732667e-02, -1.43969880e+02]),  # phi3
    np.array([ 1.85615309e-07, -3.72535276e-04])   # C_daily
]

# -----------------------------
#  Streamlit UI
# -----------------------------
st.title("The Worst Energy Model")
st.write("Enter a year to generate the energy model.")

year = st.number_input("Choose a year", min_value=1900, max_value=3000, value=2043)

# -----------------------------
#  Predict parameters for chosen year
# -----------------------------
Ay   = np.polyval(Ay_poly, year)
phiy = np.polyval(phiy_poly, year)
Cy   = np.polyval(Cy_poly, year)
daily_params = np.array([np.polyval(p, year) for p in daily_polys])

# -----------------------------
#  Build full model (hourly)
# -----------------------------
N_hours = 24 * 365
t_year = np.arange(N_hours)

# Yearly component (hourly period N)
yearly_component = yearly_sine(t_year, Ay, phiy, Cy)

# Daily component (repeat 24h cycle across the year)
t_daily = np.arange(24)
daily_cycle = daily_cosines(t_daily, *daily_params)
daily_full = np.tile(daily_cycle, N_hours // 24 + 1)[:N_hours]

# Combined model
full_model = yearly_component + daily_full

# -----------------------------
#  Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(t_year/(24*30), full_model, linewidth=0.5, color='olive')
ax.set_xlabel("Month")
ax.set_ylabel("Electricity Usage")
ax.set_title(f"Modelled Yearly Curve for {year}")
ax.grid(True)
st.pyplot(fig)

# -----------------------------
#  Export to CSV
# -----------------------------
import pandas as pd
from io import StringIO

df_export = pd.DataFrame({
    "Year": year,
    "HourIndex": t_year,                       
    "FullModel": full_model
})

st.subheader("Export")
st.write("Download the generated hourly model as a CSV file.")

# Write CSV to in-memory text buffer
csv_buffer = StringIO()
# index=False so we don’t include the DataFrame index column in the file
df_export.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode("utf-8")

st.download_button(
    label=f"Download CSV: UK_electric_grid_model_{year}.csv",
    data=csv_bytes,
    file_name=f"UK_electric_grid_model_{year}.csv",
    mime="text/csv"
)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------- 1. LOAD DATA --------
df = pd.read_csv(
    "/Users/jairajmodak/Downloads/Investment return tracking - Sheet1 (1).csv"
)

# Rename the first column to 'Month' and 'Balance' to 'Security'
df.columns = ['Month', 'Security', 'Market Value', 'Cost Basis']

# Clean the data - remove dollar signs and commas, convert to float
df['Market Value'] = df['Market Value'].str.replace('$', '').str.replace(',', '').astype(float)
df['Cost Basis'] = df['Cost Basis'].str.replace('$', '').str.replace(',', '').astype(float)

# Convert Month to datetime for proper sorting
df['Month'] = pd.to_datetime(df['Month'], format='%d/%m/%Y')

# Add Feb 2025 data
feb_2025_data = pd.DataFrame({
    'Month': pd.to_datetime(['01/02/2026'] * 9, format='%d/%m/%Y'),
    'Security': ['Cash', 'RKLB Call', 'RKLB Call', 'GOOG Call', 'NVDA', 'NVDA Call', 'PL Call', 'Databricks', 'Glean Technologies'],
    'Market Value': [25731.62, 171000.00, 147600.00, 218000.00, 99440.73, 22140.00, 97200.00, 400000.00, 399000.00],
    'Cost Basis': [25731.62, 25504.24, 63900.02, 99906.74, 59999.84, 21741.35, 70239.85, 400000.00, 301000.00]
})
df = pd.concat([df, feb_2025_data], ignore_index=True)

# Sort by Month to ensure chronological order
df = df.sort_values('Month')

# Filter to year-end dates and Feb 2026
df = df[(df['Month'].dt.month == 12) | ((df['Month'].dt.month == 2) & (df['Month'].dt.year == 2026))]

# Map ticker symbols to full company names
name_mapping = {
    'RKLB Call': 'Rocket Lab',
    'RKLB': 'Rocket Lab',
    'NVDA': 'NVIDIA',
    'NVDA Call': 'NVIDIA',
    'PL Call': 'Planet Labs',
    'PL': 'Planet Labs',
    'GOOG Call': 'Google',
    'GOOG': 'Google',
    'AMZN': 'Amazon',
    'RCAT': 'RedCat',
    'META': 'Meta',
    'MSFT': 'Microsoft',
    'AAPL': 'Apple',
    'IOT': 'Samsara',
    'CELH': 'Celsius',
    'AXON': 'Axon',
    'IGV ETF': 'IGV ETF',
    'WCLD ETF': 'WCLD ETF',
    'Glean Technologies': 'Glean Work AI',
    'Databricks': 'Databricks',
    'Cash': 'Cash',
}
df['Security'] = df['Security'].map(lambda x: name_mapping.get(x, x))

# -------- 2. CALCULATE GAIN --------
df["Unrealized Gain"] = df["Market Value"] - df["Cost Basis"]

# -------- 3. PIVOT DATA --------
cost_pivot = df.pivot_table(
    index="Month",
    columns="Security",
    values="Cost Basis",
    aggfunc="sum",
    fill_value=0
)

gain_pivot = df.pivot_table(
    index="Month",
    columns="Security",
    values="Unrealized Gain",
    aggfunc="sum",
    fill_value=0
)

# Sort the pivoted data by date
cost_pivot = cost_pivot.sort_index()
gain_pivot = gain_pivot.sort_index()

# -------- 4. PLOT STACKED BAR CHART --------
fig, ax = plt.subplots(figsize=(20, 12))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

bottom = np.zeros(len(cost_pivot))

# Custom color mapping for securities (using full names)
custom_colors = {
    'Rocket Lab': '#000000',  # Black
    'NVIDIA': '#88B730',  # Green
    'Glean Work AI': '#423CE6',  # Blue
    'Google': '#EDC03B',  # Yellow
    'Amazon': '#E57129',  # Orange
    'RedCat': '#CB3F30',  # Red
    'Planet Labs': '#539AA4',  # Teal
    'Meta': '#3C62DA',  # Blue
    'Microsoft': '#4476D2',  # Blue
    'Apple': '#A8A8A8',  # Apple Gray
    'Samsara': '#20B2AA',  # Light Sea Green
    'Cash': '#D3D3D3',  # Light Gray
    'Axon': '#FF69B4',  # Hot Pink
    'Celsius': '#FA8072',  # Salmon
    'IGV ETF': '#DDA0DD',  # Plum
    'WCLD ETF': '#9ACD32',  # Yellow Green
    'Databricks': '#FF3621',  # Databricks Red
}

# Default colors for any securities not in custom mapping
default_colors = plt.cm.tab20.colors

# Format x-axis labels as dates
x_labels = [date.strftime('%b %Y') for date in cost_pivot.index]
x_positions = np.arange(len(x_labels))

# Calculate totals for each period
period_totals = cost_pivot.sum(axis=1) + gain_pivot.sum(axis=1)

# Calculate MARKET VALUE for each security
market_value_pivot = cost_pivot + gain_pivot

# Helper function to determine contrasting text color
def get_contrast_color(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = r * alpha + 255 * (1 - alpha), g * alpha + 255 * (1 - alpha), b * alpha + 255 * (1 - alpha)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return 'black' if luminance > 0.5 else 'white'

# Plot each time period
for period_idx, period in enumerate(market_value_pivot.index):
    period_values = market_value_pivot.loc[period]
    period_costs = cost_pivot.loc[period]
    period_gains = gain_pivot.loc[period]

    # Sort THIS period's securities by value (largest first, so largest is at bottom)
    sorted_period_securities = period_values[period_values > 0].sort_values(ascending=False).index

    current_bottom = 0

    # First pass: plot all cost blocks at bottom
    for security in sorted_period_securities:
        cost_value = period_costs[security] if security in period_costs.index else 0

        if cost_value > 0:
            base_color = custom_colors.get(security, default_colors[list(market_value_pivot.columns).index(security) % len(default_colors)])

            ax.bar(
                x_positions[period_idx],
                cost_value,
                bottom=current_bottom,
                color=base_color,
                alpha=0.9,
                edgecolor='black',
                linewidth=0.3,
                width=0.9
            )

            # Add cost label (name only, no figures)
            if cost_value > 30000:
                text_color = get_contrast_color(base_color if isinstance(base_color, str) else '#{:02x}{:02x}{:02x}'.format(int(base_color[0]*255), int(base_color[1]*255), int(base_color[2]*255)), 0.9)
                label = f"{security}"
                ax.text(x_positions[period_idx], current_bottom + cost_value/2,
                       label, ha='center', va='center', fontsize=16,
                       color=text_color, fontweight='normal', rotation=0)

            # Add horizontal separator line
            ax.hlines(current_bottom + cost_value, x_positions[period_idx] - 0.43, x_positions[period_idx] + 0.43,
                     colors='black', linewidth=0.5, alpha=0.4)

            current_bottom += cost_value

    # Second pass: plot all gain blocks on top (dark green)
    for security in sorted_period_securities:
        gain_value = period_gains[security] if security in period_gains.index else 0

        if gain_value > 0:
            base_color = custom_colors.get(security, default_colors[list(market_value_pivot.columns).index(security) % len(default_colors)])

            ax.bar(
                x_positions[period_idx],
                gain_value,
                bottom=current_bottom,
                color='#228B22',
                alpha=0.9,
                edgecolor='black',
                linewidth=0.3,
                width=0.9
            )

            # Add gain label (name only, no figures)
            if gain_value > 30000:
                text_color = get_contrast_color('#228B22', 0.9)
                label = f"{security}"
                ax.text(x_positions[period_idx], current_bottom + gain_value/2,
                       label, ha='center', va='center', fontsize=16,
                       color=text_color, fontweight='normal', rotation=0)

            # Add horizontal separator line
            ax.hlines(current_bottom + gain_value, x_positions[period_idx] - 0.43, x_positions[period_idx] + 0.43,
                     colors='black', linewidth=0.5, alpha=0.4)

            current_bottom += gain_value

# -------- 5. STYLING --------
ax.set_title("Portfolio composition over time", fontsize=26)
ax.set_ylabel("")
ax.set_xlabel("Year", fontsize=18)

# Hide y-axis values
ax.set_yticklabels([])
ax.tick_params(axis='y', length=0)

# Set x-axis labels
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=20)

# Add 'Current' label below the last x-tick
ax.text(x_positions[-1], -0.07, 'Current', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=20, fontweight='bold', color='#1a53ff')
ax.tick_params(axis='y', labelsize=20)

plt.tight_layout()
plt.savefig('/Users/jairajmodak/Kevlar-Capital/portfolio_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n--- Portfolio Summary ---")
print(f"Total unique time periods: {len(df['Month'].unique())}")
print(f"Total securities tracked: {len(df['Security'].unique())}")
print(f"\nTotal Portfolio Value by Period:")
portfolio_totals = df.groupby('Month').agg({
    'Cost Basis': 'sum',
    'Market Value': 'sum',
    'Unrealized Gain': 'sum'
}).round(2)
print(portfolio_totals)
print(f"\nOverall Total Cost Basis: ${df['Cost Basis'].sum():,.2f}")
print(f"Overall Total Market Value: ${df['Market Value'].sum():,.2f}")
print(f"Overall Total Unrealized Gain: ${df['Unrealized Gain'].sum():,.2f}")
print(f"Overall Return: {(df['Unrealized Gain'].sum() / df['Cost Basis'].sum() * 100):.2f}%")
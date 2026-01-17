import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_aggregate_datasets():
    print("Initializing standalone data generation...")
    np.random.seed(42)

    # ==========================================
    # 1. SETUP: LISTS & DATA ASSETS
    # ==========================================

    # 150+ Ontario FSAs
    ontario_fsas = [
        'M1B', 'M1C', 'M1E', 'M1G', 'M1H', 'M1J', 'M1K', 'M1L', 'M1M', 'M1N',
        'M1P', 'M1R', 'M1S', 'M1T', 'M1V', 'M1W', 'M1X', 'M2H', 'M2J', 'M2K',
        'M2L', 'M2M', 'M2N', 'M2P', 'M2R', 'M3A', 'M3B', 'M3C', 'M3H', 'M3J',
        'M3K', 'M3L', 'M3M', 'M3N', 'M4A', 'M4B', 'M4C', 'M4E', 'M4G', 'M4H',
        'M4J', 'M4K', 'M4L', 'M4M', 'M4N', 'M4P', 'M4R', 'M4S', 'M4T', 'M4V',
        'M4W', 'M4X', 'M4Y', 'M5A', 'M5B', 'M5C', 'M5E', 'M5G', 'M5H', 'M5J',
        'M5K', 'M5L', 'M5M', 'M5N', 'M5P', 'M5R', 'M5S', 'M5T', 'M5V', 'M5W',
        'M5X', 'M6A', 'M6B', 'M6C', 'M6E', 'M6G', 'M6H', 'M6J', 'M6K', 'M6L',
        'M6M', 'M6N', 'M6P', 'M6R', 'M6S', 'M7A', 'M8V', 'M8W', 'M8X', 'M8Y',
        'M8Z', 'M9A', 'M9B', 'M9C', 'M9L', 'M9M', 'M9N', 'M9P', 'M9R', 'M9V', 'M9W',
        'L1S', 'L1T', 'L1V', 'L3R', 'L3T', 'L4B', 'L4C', 'L4J', 'L4K', 'L4L',
        'L4T', 'L4W', 'L4X', 'L4Y', 'L4Z', 'L5A', 'L5B', 'L5C', 'L5E', 'L5G',
        'L5H', 'L5J', 'L5K', 'L5L', 'L5M', 'L5N', 'L5R', 'L5V', 'L5W', 'L6A',
        'L6B', 'L6C', 'L6E', 'L6G', 'L6H', 'L6J', 'L6K', 'L6L', 'L6M', 'L6P',
        'L6R', 'L6S', 'L6T', 'L6V', 'L6W', 'L6X', 'L6Y', 'L6Z', 'L7A', 'L7B',
        'K1A', 'K1B', 'K1C', 'K1E', 'K1G', 'K1H', 'K1J', 'K1K', 'K1L', 'K1M',
        'K1N', 'K1P', 'K1R', 'K1S', 'K1T', 'K1V', 'K1W', 'K1X', 'K1Y', 'K1Z',
        'K2A', 'K2B', 'K2C', 'K2E', 'K2G', 'K2H', 'K2J', 'K2K', 'K2L', 'K2M',
        'N1H', 'N1K', 'N2L', 'N2T', 'N5V', 'N6A', 'N8X', 'N9A', 'P3E', 'P6A'
    ]

    # 100+ Vehicle Codes
    vehicles = [
        ('Toyota', 'Corolla'), ('Toyota', 'Camry'), ('Toyota', 'RAV4'), ('Toyota', 'Highlander'),
        ('Toyota', '4Runner'), ('Toyota', 'Tacoma'), ('Toyota', 'Tundra'), ('Toyota', 'Sienna'),
        ('Toyota', 'Prius'), ('Toyota', 'Venza'), ('Toyota', 'Crown'), ('Toyota', 'Sequoia'),
        ('Honda', 'Civic'), ('Honda', 'Accord'), ('Honda', 'CR-V'), ('Honda', 'Pilot'),
        ('Honda', 'Odyssey'), ('Honda', 'HR-V'), ('Honda', 'Passport'), ('Honda', 'Ridgeline'),
        ('Lexus', 'RX350'), ('Lexus', 'NX300'), ('Lexus', 'IS300'), ('Lexus', 'ES350'),
        ('Lexus', 'UX250h'), ('Lexus', 'GX460'), ('Lexus', 'LX600'), ('Lexus', 'RC350'),
        ('Ford', 'F-150'), ('Ford', 'Escape'), ('Ford', 'Explorer'), ('Ford', 'Edge'),
        ('Ford', 'Mustang'), ('Ford', 'Bronco'), ('Ford', 'Ranger'), ('Ford', 'Expedition'),
        ('Land Rover', 'Range Rover Sport'), ('Land Rover', 'Range Rover Evoque'),
        ('Land Rover', 'Defender'), ('Land Rover', 'Velar'), ('Land Rover', 'Discovery'),
        ('Dodge', 'Ram 1500'), ('Dodge', 'Durango'), ('Dodge', 'Charger'), ('Dodge', 'Challenger'),
        ('Jeep', 'Wrangler'), ('Jeep', 'Grand Cherokee'), ('Jeep', 'Cherokee'), ('Jeep', 'Compass'),
        ('Jeep', 'Gladiator'), ('Jeep', 'Wagoneer'),
        ('BMW', '3 Series'), ('BMW', '5 Series'), ('BMW', 'X1'), ('BMW', 'X3'), ('BMW', 'X5'),
        ('BMW', 'X7'), ('Mercedes', 'C-Class'), ('Mercedes', 'E-Class'), ('Mercedes', 'GLC'),
        ('Mercedes', 'GLE'), ('Mercedes', 'GLA'), ('Mercedes', 'G-Wagon'),
        ('Audi', 'A4'), ('Audi', 'Q3'), ('Audi', 'Q5'), ('Audi', 'Q7'),
        ('Hyundai', 'Elantra'), ('Hyundai', 'Tucson'), ('Hyundai', 'Santa Fe'), ('Hyundai', 'Kona'),
        ('Hyundai', 'Palisade'), ('Hyundai', 'Ioniq 5'),
        ('Kia', 'Sportage'), ('Kia', 'Seltos'), ('Kia', 'Forte'), ('Kia', 'Telluride'),
        ('Kia', 'Sorento'), ('Kia', 'Soul'),
        ('Subaru', 'Crosstrek'), ('Subaru', 'Outback'), ('Subaru', 'Forester'),
        ('Mazda', 'CX-5'), ('Mazda', 'CX-30'), ('Mazda', 'Mazda3'),
        ('Volkswagen', 'Tiguan'), ('Volkswagen', 'Jetta'), ('Volkswagen', 'Atlas'),
        ('Nissan', 'Rogue'), ('Nissan', 'Sentra'), ('Nissan', 'Pathfinder'),
        ('Chevrolet', 'Silverado'), ('Chevrolet', 'Equinox'), ('Chevrolet', 'Tahoe'),
        ('Tesla', 'Model 3'), ('Tesla', 'Model Y')
    ]

    # Flatten vehicle tuple to string code
    veh_codes_list = [f"{m} {model}" for m, model in vehicles]

    # ==========================================
    # 2. GENERATE BASE COMPANY DATA (INTERNAL)
    # ==========================================
    n_rows = 50000
    print(f"Generating base company data ({n_rows} rows)...")

    # Assign FSA and Vehicle randomly to company policies
    fsa_col = np.random.choice(ontario_fsas, n_rows)
    veh_col = np.random.choice(veh_codes_list, n_rows)

    # Calculate Exposure (Random duration between 0 and 1 year)
    # Logic: Start date random in last 2 years, expiry 1 year later.
    start_dates = [datetime.today() - timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)]
    expiry_dates = [d + timedelta(days=365) for d in start_dates]
    today = datetime.today()

    # Vectorized exposure calc
    start_series = pd.Series(start_dates)
    expiry_series = pd.Series(expiry_dates)
    effective_end = expiry_series.apply(lambda x: min(x, today))
    exposure_days = (effective_end - start_series).dt.days
    exposure_col = (exposure_days / 365.0).clip(0, 1.0)

    # Create Base DataFrame
    df_base = pd.DataFrame({
        'garaging postal code FSA': fsa_col,
        'vehice code': veh_col,
        'earned exposure': exposure_col
    })

    # ==========================================
    # 3. DEFINE GLOBAL TOTALS
    # ==========================================
    # These are the "Truth" numbers that both datasets must match.

    TOTAL_COMPANY_EXPOSURE = df_base['earned exposure'].sum()

    # We simulate that the company has ~2% market share.
    # Total Market Vehicles = Company Count / 0.02
    TOTAL_MARKET_VEHICLES = int(len(df_base) / 0.02)  # ~2.5 Million

    print(f"Target Company Exposure Sum: {TOTAL_COMPANY_EXPOSURE:,.2f}")
    print(f"Target Market Vehicle Sum:   {TOTAL_MARKET_VEHICLES:,}")

    # ==========================================
    # 4. CREATE FSA DATASET
    # ==========================================
    print("Building FSA Aggregates...")

    # A. Company Exposure Sum per FSA
    df_fsa = df_base.groupby('garaging postal code FSA')['earned exposure'].sum().reset_index()
    df_fsa.rename(columns={'earned exposure': 'Company_Exposure'}, inplace=True)

    # B. Generate Market Vehicle Counts per FSA
    # Logic: Market count is proportional to Company Exposure + Random Noise
    # This simulates that the company might be over/under-indexed in certain FSAs
    np.random.seed(101)
    weights_fsa = df_fsa['Company_Exposure'] / df_fsa['Company_Exposure'].sum()
    weights_fsa = weights_fsa * np.random.uniform(0.7, 1.3, len(weights_fsa))  # Add variance
    weights_fsa = weights_fsa / weights_fsa.sum()  # Re-normalize to 1.0

    df_fsa['Total_Vehicles_Market'] = (weights_fsa * TOTAL_MARKET_VEHICLES).astype(int)

    # Balance check (rounding errors will cause slight mismatch, we force exact match)
    diff_fsa = TOTAL_MARKET_VEHICLES - df_fsa['Total_Vehicles_Market'].sum()
    df_fsa.loc[0, 'Total_Vehicles_Market'] += diff_fsa  # Add remainder to first row

    # ==========================================
    # 5. CREATE VEHICLE DATASET
    # ==========================================
    print("Building Vehicle Aggregates...")

    # A. Company Exposure Sum per Vehicle
    df_veh = df_base.groupby('vehice code')['earned exposure'].sum().reset_index()
    df_veh.rename(columns={'earned exposure': 'Company_Exposure'}, inplace=True)

    # B. Generate Market Vehicle Counts per Vehicle
    np.random.seed(202)
    weights_veh = df_veh['Company_Exposure'] / df_veh['Company_Exposure'].sum()
    weights_veh = weights_veh * np.random.uniform(0.6, 1.4, len(weights_veh))  # Add variance
    weights_veh = weights_veh / weights_veh.sum()  # Re-normalize

    df_veh['Total_Vehicles_Market'] = (weights_veh * TOTAL_MARKET_VEHICLES).astype(int)

    # Balance check
    diff_veh = TOTAL_MARKET_VEHICLES - df_veh['Total_Vehicles_Market'].sum()
    df_veh.loc[0, 'Total_Vehicles_Market'] += diff_veh

    # ==========================================
    # 6. VALIDATION & RETURN
    # ==========================================
    print("\n--- Final Validation ---")

    exp_fsa = df_fsa['Company_Exposure'].sum()
    exp_veh = df_veh['Company_Exposure'].sum()
    print(f"Company Exposure (FSA): {exp_fsa:,.2f}")
    print(f"Company Exposure (Veh): {exp_veh:,.2f}")
    print(f"MATCH: {np.isclose(exp_fsa, exp_veh)}")

    mkt_fsa = df_fsa['Total_Vehicles_Market'].sum()
    mkt_veh = df_veh['Total_Vehicles_Market'].sum()
    print(f"Market Vehicles (FSA):  {mkt_fsa:,}")
    print(f"Market Vehicles (Veh):  {mkt_veh:,}")
    print(f"MATCH: {mkt_fsa == mkt_veh}")

    return df_fsa, df_veh


if __name__ == "__main__":
    fsa_data, veh_data = generate_aggregate_datasets()

    # Export to CSV
    fsa_data.to_csv("aggregate_fsa_data.csv", index=False)
    veh_data.to_csv("aggregate_vehicle_data.csv", index=False)
    print("\nFiles 'aggregate_fsa_data.csv' and 'aggregate_vehicle_data.csv' generated successfully.")
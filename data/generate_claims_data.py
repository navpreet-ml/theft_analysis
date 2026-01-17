import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_clean_ontario_data(n_rows=50000):
    print(f"Generating {n_rows:,} rows of Ontario data with strictly requested columns...")
    np.random.seed(42)

    # ==========================================
    # 1. DATA ASSETS (150+ FSAs, 115+ Vehicles)
    # ==========================================

    # 150+ Ontario FSAs (Toronto, GTA, Ottawa, Western/Northern ON)
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

    # 100+ Vehicle Models (Make, Model, Base Price)
    vehicles = [
        ('Toyota', 'Corolla', 26000), ('Toyota', 'Camry', 34000), ('Toyota', 'RAV4', 38000),
        ('Toyota', 'Highlander', 52000), ('Toyota', '4Runner', 58000), ('Toyota', 'Tacoma', 48000),
        ('Toyota', 'Tundra', 65000), ('Toyota', 'Sienna', 50000), ('Toyota', 'Prius', 36000),
        ('Toyota', 'Venza', 42000), ('Toyota', 'Crown', 55000), ('Toyota', 'Sequoia', 85000),
        ('Honda', 'Civic', 28000), ('Honda', 'Accord', 35000), ('Honda', 'CR-V', 39000),
        ('Honda', 'Pilot', 55000), ('Honda', 'Odyssey', 51000), ('Honda', 'HR-V', 31000),
        ('Honda', 'Passport', 48000), ('Honda', 'Ridgeline', 52000),
        ('Lexus', 'RX350', 65000), ('Lexus', 'NX300', 52000), ('Lexus', 'IS300', 48000),
        ('Lexus', 'ES350', 55000), ('Lexus', 'UX250h', 44000), ('Lexus', 'GX460', 75000),
        ('Lexus', 'LX600', 110000), ('Lexus', 'RC350', 58000),
        ('Ford', 'F-150', 60000), ('Ford', 'Escape', 33000), ('Ford', 'Explorer', 50000),
        ('Ford', 'Edge', 40000), ('Ford', 'Mustang', 45000), ('Ford', 'Bronco', 55000),
        ('Ford', 'Ranger', 42000), ('Ford', 'Expedition', 78000), ('Ford', 'Maverick', 35000),
        ('Land Rover', 'Range Rover Sport', 115000), ('Land Rover', 'Range Rover Evoque', 62000),
        ('Land Rover', 'Defender', 85000), ('Land Rover', 'Velar', 72000),
        ('Land Rover', 'Discovery', 80000),
        ('Dodge', 'Ram 1500', 62000), ('Dodge', 'Durango', 55000), ('Dodge', 'Charger', 45000),
        ('Dodge', 'Challenger', 48000), ('Jeep', 'Wrangler', 50000), ('Jeep', 'Grand Cherokee', 60000),
        ('Jeep', 'Cherokee', 40000), ('Jeep', 'Compass', 35000), ('Jeep', 'Gladiator', 58000),
        ('Jeep', 'Wagoneer', 85000),
        ('BMW', '3 Series', 55000), ('BMW', '5 Series', 70000), ('BMW', 'X1', 45000),
        ('BMW', 'X3', 60000), ('BMW', 'X5', 85000), ('BMW', 'X7', 105000),
        ('Mercedes', 'C-Class', 58000), ('Mercedes', 'E-Class', 75000), ('Mercedes', 'GLC', 62000),
        ('Mercedes', 'GLE', 88000), ('Mercedes', 'GLA', 48000), ('Mercedes', 'G-Wagon', 180000),
        ('Audi', 'A4', 52000), ('Audi', 'Q3', 46000), ('Audi', 'Q5', 60000), ('Audi', 'Q7', 82000),
        ('Hyundai', 'Elantra', 24000), ('Hyundai', 'Tucson', 34000), ('Hyundai', 'Santa Fe', 42000),
        ('Hyundai', 'Kona', 29000), ('Hyundai', 'Palisade', 54000), ('Hyundai', 'Ioniq 5', 55000),
        ('Kia', 'Sportage', 33000), ('Kia', 'Seltos', 28000), ('Kia', 'Forte', 23000),
        ('Kia', 'Telluride', 56000), ('Kia', 'Sorento', 45000), ('Kia', 'Soul', 26000),
        ('Subaru', 'Crosstrek', 32000), ('Subaru', 'Outback', 38000), ('Subaru', 'Forester', 36000),
        ('Mazda', 'CX-5', 36000), ('Mazda', 'CX-30', 30000), ('Mazda', 'Mazda3', 28000),
        ('Volkswagen', 'Tiguan', 38000), ('Volkswagen', 'Jetta', 27000), ('Volkswagen', 'Atlas', 50000),
        ('Nissan', 'Rogue', 35000), ('Nissan', 'Sentra', 25000), ('Nissan', 'Pathfinder', 48000),
        ('Chevrolet', 'Silverado', 58000), ('Chevrolet', 'Equinox', 32000), ('Chevrolet', 'Tahoe', 75000),
        ('Tesla', 'Model 3', 55000), ('Tesla', 'Model Y', 65000)
    ]

    # ==========================================
    # 2. GENERATION LOGIC
    # ==========================================

    # Vectorized Indexing for speed
    veh_indices = np.random.randint(0, len(vehicles), n_rows)
    fsa_indices = np.random.randint(0, len(ontario_fsas), n_rows)

    data = {
        'source system': np.random.choice(['Legacy', 'Guidewire'], n_rows, p=[0.3, 0.7]),
        'underwriting company name': np.random.choice(['Aviva', 'Intact', 'Desjardins', 'TD', 'Travelers'], n_rows),
        'policy number': [f"POL-ON-{x}" for x in np.random.randint(1000000, 9999999, n_rows)],
        'risk number': np.random.randint(1, 4, n_rows),
        'driver type': np.random.choice(['Primary', 'Occasional', 'Young', 'Senior'], n_rows, p=[0.6, 0.2, 0.1, 0.1]),
        'episode start date': [datetime.today() - timedelta(days=np.random.randint(0, 730)) for _ in range(n_rows)],
        'annual km driven': np.random.randint(5000, 35000, n_rows),
        'garaging postal code FSA': [ontario_fsas[i] for i in fsa_indices],
        'province': 'ON',
        'vehicle make': [vehicles[i][0] for i in veh_indices],
        'model': [vehicles[i][1] for i in veh_indices],
        'vehicle year': np.random.randint(2019, 2026, n_rows),
        'tag discount': np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
    }

    df = pd.DataFrame(data)

    # Derived Columns
    df['vehice code'] = df['vehicle make'] + ' ' + df['model']
    df['episode expiry date'] = df['episode start date'] + timedelta(days=365)

    # Price Logic (Base Price from Tuple + Random Variance + Depreciation)
    base_prices = np.array([vehicles[i][2] for i in veh_indices])
    df['price of new car'] = base_prices * np.random.uniform(0.98, 1.15, n_rows)

    age = 2026 - df['vehicle year']
    depr_factor = np.maximum(0.55, 1 - (age * 0.08))
    df['purchase amount'] = df['price of new car'] * depr_factor

    # VIN Generation (Mock)
    df['vehicle vin number'] = '2' + df['vehicle make'].str[:2].str.upper() + \
                               pd.Series(np.random.randint(10000, 99999, n_rows)).astype(str) + \
                               'ON' + pd.Series(np.random.randint(100000, 999999, n_rows)).astype(str)

    # Exposure Logic
    today = datetime.today()
    df['effective_end'] = df['episode expiry date'].apply(lambda x: min(x, today))
    df['earned exposure'] = (df['effective_end'] - df['episode start date']).dt.days / 365.0
    df['earned exposure'] = df['earned exposure'].clip(0, 1.0)

    # ==========================================
    # 3. HIDDEN THEFT LOGIC
    # ==========================================

    # We calculate probabilities internally to generate realistic data,
    # but we DO NOT return the probability column.

    internal_prob = np.full(n_rows, 0.012)  # 1.2% base

    # High Risk Vehicles
    high_risk_makes = ['Land Rover', 'Lexus', 'Dodge', 'Toyota', 'Jeep']
    is_high_risk = df['vehicle make'].isin(high_risk_makes)
    internal_prob[is_high_risk] += 0.025

    # High Risk Locations (Peel/Toronto)
    risky_fsas = ['L4T', 'L5N', 'M9V', 'M1B', 'L6P']
    is_risky_geo = df['garaging postal code FSA'].isin(risky_fsas)
    internal_prob[is_risky_geo] += 0.02

    # TAG Discount reduces risk
    internal_prob[df['tag discount'] == 1] *= 0.4

    # Determine Theft
    rng = np.random.random(n_rows)
    df['claim count theft'] = np.where(rng < internal_prob, 1, 0)

    # Calculate Incurred Amount (0 if no theft)
    df['incurred theft claim'] = 0.0

    # Logic: 70% Total Loss, 30% Partial
    sev_rng = np.random.random(n_rows)
    claims_mask = df['claim count theft'] == 1

    # Total Loss
    total_mask = claims_mask & (sev_rng < 0.7)
    df.loc[total_mask, 'incurred theft claim'] = df.loc[total_mask, 'purchase amount'] * np.random.uniform(0.9, 1.1)

    # Partial Loss
    partial_mask = claims_mask & (sev_rng >= 0.7)
    df.loc[partial_mask, 'incurred theft claim'] = np.random.uniform(1500, 8500, partial_mask.sum())

    # Loss Date (Only for claims)
    df['loss date'] = pd.NaT
    offsets = np.random.randint(1, 180, claims_mask.sum())
    df.loc[claims_mask, 'loss date'] = df.loc[claims_mask, 'episode start date'] + pd.to_timedelta(offsets, unit='D')

    # ==========================================
    # 4. FINAL CLEANUP
    # ==========================================

    # Final selection of columns strictly as requested
    final_columns = [
        'source system', 'underwriting company name', 'policy number', 'risk number',
        'driver type', 'episode start date', 'episode expiry date', 'annual km driven',
        'garaging postal code FSA', 'province', 'purchase amount', 'price of new car',
        'vehicle make', 'model', 'vehicle vin number', 'vehicle year', 'vehice code',
        'loss date', 'claim count theft', 'incurred theft claim', 'earned exposure',
        'tag discount'
    ]

    df_final = df[final_columns]

    print("Generation complete. Returning clean DataFrame.")
    return df_final

# USAGE:
df = generate_clean_ontario_data(50000)
df.to_csv("clean_ontario_theft_data.csv", index=False)

# Load the interpolated results

# Find the license plate with the highest confidence for each vehicle

# Infer vehicle type from car_id (update logic as needed)
def infer_vehicle_type(car_id):
    try:
        cid = int(car_id)
        if 1 <= cid < 100:
            return 'car'
        elif 100 <= cid < 200:
            return 'bus'
        elif 200 <= cid < 300:
            return 'truck'
        else:
            return 'unknown'
    except:
        return 'unknown'


# Save to CSV

import sys
if __name__ == "__main__":
    import pandas as pd
    in_csv = sys.argv[1] if len(sys.argv) > 1 else 'test_interpolated.csv'
    out_csv = sys.argv[2] if len(sys.argv) > 2 else 'unique_vehicles.csv'
    results = pd.read_csv(in_csv)
    idx = results.groupby('car_id')['license_number_score'].idxmax()
    unique_vehicles = results.loc[idx, ['car_id', 'license_number', 'license_number_score']].copy()
    def infer_vehicle_type(car_id):
        try:
            cid = int(car_id)
            if 1 <= cid < 100:
                return 'car'
            elif 100 <= cid < 200:
                return 'bus'
            elif 200 <= cid < 300:
                return 'truck'
            else:
                return 'unknown'
        except:
            return 'unknown'
    unique_vehicles['vehicle_type'] = unique_vehicles['car_id'].apply(infer_vehicle_type)
    unique_vehicles.to_csv(out_csv, index=False)
    print(f'Filtered unique vehicles saved to {out_csv}')

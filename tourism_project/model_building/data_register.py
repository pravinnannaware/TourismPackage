import os
import pandas as pd

def register_data():
    data_path = "tourism_project/data/tourism.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")

    df = pd.read_csv(data_path)
    
    expected_columns = [
        "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
        "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
        "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
        "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
        "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
    ]
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
        
    print("=== Data Schema Validation Passed ===")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\nClass Distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True))

if __name__ == "__main__":
    register_data()

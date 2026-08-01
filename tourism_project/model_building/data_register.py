import os
import pandas as pd

def register_data():
    """
    Validates the dataset file existence, schema integrity, and prints
    a high-level summary of the dataset for the pipeline execution.
    """
    # Define the relative path to the dataset within the project directory
    data_path = "tourism_project/data/tourism.csv"

    # 1. Verify file presence
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")

    df = pd.read_csv(data_path)

    # 2. Define the expected column schema required for model training
    expected_columns = [
        "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
        "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
        "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
        "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
        "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
    ]
    
    # Check for any missing columns against the expected schema
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # 3. Output dataset validation summary and target class distribution    
    print("=== Data Schema Validation Passed ===")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\nClass Distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True))

if __name__ == "__main__":
  # Execute registration logic when script is run directly
    register_data()

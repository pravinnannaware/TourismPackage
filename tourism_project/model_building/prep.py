import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    
    """
    Loads raw tourism dataset, performs basic cleaning and imputation,
    splits the data into stratified train/test sets, and exports CSV artifacts.
    """
    # 1. Load the registered dataset from the project directory

    df = pd.read_csv("tourism_project/data/tourism.csv")
    
    # 2. Drop unnecessary index and unique identifier columns
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        
    # Drop customer ID column
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)
        
    # 3. Standardize categorical typos (e.g., 'Fe Male' -> 'Female')
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    
    # 4. Handle Missing Values (Imputation)
    # Impute missing numerical features with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if col != "ProdTaken":
            df[col] = df[col].fillna(df[col].median())
            
    # Impute missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Separate features (X) and target variable (y)
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # 6. Perform stratified Train-Test Split (80% Train, 20% Test)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Save train/test datasets locally as CSV artifacts for downstream training
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete. Artifacts saved locally.")

if __name__ == "__main__":
    # Execute data preparation logic when script is run directly  
    prepare_data()

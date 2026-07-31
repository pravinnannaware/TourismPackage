import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    df = pd.read_csv("data/Tourism.csv")
    
    # Drop unique ID column
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)
        
    # Standardize categorical values (e.g., Fe Male -> Female)
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    
    # Impute missing numerical values with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if col != "ProdTaken":
            df[col] = df[col].fillna(df[col].median())
            
    # Impute missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete. Artifacts saved locally.")

if __name__ == "__main__":
    prepare_data()

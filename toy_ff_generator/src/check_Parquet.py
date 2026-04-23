import pandas as pd

file_path = r"C:\Users\Yuan\Desktop\YUAN\實習\北京量化實習\FactorMarketRL\toy_ff_generator\outputs\data v3\bull\bull_10_200_PL_1.parquet"

df = pd.read_parquet(file_path)

print("=== DataFrame shape ===")
print(df.shape)

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Head ===")
print(df.head())

print("\n=== Full DataFrame ===")
print(df)

print("\n=== Full DataFrame ===")
print(df['mu'])
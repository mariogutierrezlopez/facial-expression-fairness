import pandas as pd

TRAIN_CSV = "/home12TB1/database/recognition/faces/affwild2/dataframe_train.csv"

train_df = pd.read_csv(TRAIN_CSV)

print(f"La longitud del csv es de {len(train_df)}")
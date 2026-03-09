import pandas as pd

# PATH_TRAIN_CSV = "/home12TB1/database/recognition/faces/RAF-DB/rafdb_train.csv"
PATH_TRAIN_CSV = "/home12TB1/database/recognition/faces/RAF-DB/rafdb_test_pose_quality_fft.csv"
PATH_IMG2POSE_CSV = "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/train.csv"
OUTPUT = "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/test_complete.csv"

def merge_datasets():
    print("Juntando datasets")
    df_train = pd.read_csv(PATH_TRAIN_CSV)
    df_i2p = pd.read_csv(PATH_IMG2POSE_CSV)

    print(f"Filas archivo train: {len(df_train)}")
    print(f"Filas archivo con anotaciones img2pose: {len(df_i2p)}")

    df_final = pd.merge(df_train, df_i2p, on='path', how='inner')

    print(f"Filas finales: {len(df_final)}")

    df_final.to_csv(OUTPUT, index=False)
    print(f"El dataset se ha guardado correctamente en {OUTPUT}")

if __name__ == "__main__":
    merge_datasets()
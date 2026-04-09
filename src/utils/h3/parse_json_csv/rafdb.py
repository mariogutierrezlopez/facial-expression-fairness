import os
import json
import pandas as pd
from tqdm import tqdm
import re
import traceback

BASE_PATH = "/home12TB1/database/recognition/faces/affectnet"
RAFDB_TRAIN_PATH = "/home12TB1/database/recognition/faces/RAF-DB/Image/aligned"

JSON_DIR_TRAIN = "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/json_results"


def migrate_json_to_csv(root_dir, output_csv):
    data_list = []

    print(f"Buscando archivos en el directorio {root_dir}")
    for subdir, dirs, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.endswith(".json"):
                file_path = os.path.join(subdir, file)

                with open(file_path, 'r') as f:
                    try:
                        content = json.load(f)

                        #Diccionarios principales
                        i2p_list = content.get("img2pose_data") or []
                        i2p = i2p_list[0] if (isinstance(i2p_list, list) and len(i2p_list) > 0) else {}
                        row = {
                            "path": _get_path(file_path),
                            "i2p_score": i2p.get("score"),
                            "i2p_pitch": i2p.get("pitch"),
                            "i2p_yaw": i2p.get("yaw"),
                            "i2p_roll": i2p.get("roll")
                        }
                        data_list.append(row)
                    except Exception as e:
                        print(f"Error {e} en el archivo: {file_path}")
                        traceback.print_exc()
                        print("-" * 60)
    
    df = pd.DataFrame(data_list)

    df.to_csv(output_csv, index=False)
    print(f"Procesos terminado, se han guardado {len(df)} filas en {output_csv}")

# A partir de la ruta del json de anotaciones obtengo la ruta de la imagen
def _get_path(json_path): 
    
    img_path = ""
    json_file = os.path.basename(json_path)
    img_name = json_file.replace('.json', '.jpg')
    img_path = os.path.join(RAFDB_TRAIN_PATH, img_name)
    
    return img_path

if __name__ == "__main__":

    print("Parseando archivos")
    output_csv = "/home12TB1/database/recognition/faces/RAF-DB/img2pose_ann/train.csv"
    migrate_json_to_csv(JSON_DIR_TRAIN, output_csv)
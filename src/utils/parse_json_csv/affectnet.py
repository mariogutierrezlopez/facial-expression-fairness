import os
import json
import pandas as pd
from tqdm import tqdm
import re
import traceback

BASE_PATH = "/home12TB1/database/recognition/faces/affectnet"
AFFECT_NET_TRAIN_PATH = "/home12TB1/database/recognition/faces/affectnet/human_annotated/train_set/images"
AFFECTNET_VAL_PATH = "/home12TB1/database/recognition/faces/affectnet/human_annotated/validation_set/images"
AFFECTNET_NO_HUMAN_ANNOTATED_PATH = "home12TB1/database/recognition/faces/affectnet/no_human_annotated/images"

JSON_DIR_TRAIN = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/train/json_results"
JSON_DIR_VAL = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/validation/json_results"
JSON_DIR_NOHUMANANNOTATED = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/no_human_annotated/json_results"


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

                        softlabel = content.get("soft-label") or []
                        metadata = content.get("meta-data") or {}
                        gender = metadata.get("gender") or {}
                        race = metadata.get("race") or {}
                        landmarks_68 = metadata.get("landmark-68") or []

                        i2p_list = content.get("img2pose_data") or []
                        i2p = i2p_list[0] if (isinstance(i2p_list, list) and len(i2p_list) > 0) else {}


                        row = {
                            "path": _get_path(file_path),
                            "human_label": content.get("human-label"),
                            "soft_0": softlabel[0] if len(softlabel) > 0 else None,
                            "soft_1": softlabel[1] if len(softlabel) > 1 else None,
                            "soft_2": softlabel[2] if len(softlabel) > 2 else None,
                            "soft_3": softlabel[3] if len(softlabel) > 3 else None,
                            "soft_4": softlabel[4] if len(softlabel) > 4 else None,
                            "soft_5": softlabel[5] if len(softlabel) > 5 else None,
                            "soft_6": softlabel[6] if len(softlabel) > 6 else None,
                            "soft_7": softlabel[7] if len(softlabel) > 7 else None,
                            "subset": content.get("subset"),
                            "age": metadata.get("age"),
                            "valence": metadata.get("valence"),
                            "arousal": metadata.get("arousal"),
                            "gender_female": gender.get("female"),
                            "gender_male": gender.get("male"),
                            "race_asian": race.get("asian"),
                            "race_indian": race.get("indian"),
                            "race_black": race.get("black"),
                            "race_white": race.get("white"),
                            "race_middle_eastern": race.get("middle eastern"),
                            "race_latino_hispanic": race.get("latino hispanic"),
                            "i2p_score": i2p.get("score"),
                            "i2p_pitch": i2p.get("pitch"),
                            "i2p_yaw": i2p.get("yaw"),
                            "i2p_roll": i2p.get("roll")
                        }

                        #landmarks 68
                        if landmarks_68 and len(landmarks_68) == 136:
                            for i in range(68):
                                row[f"x{i}"] = landmarks_68[i * 2]
                                row[f"y{i}"] = landmarks_68[i * 2 + 1]
                        else:
                            for i in range(68):
                                row[f"x{i}"] = None
                                row[f"y{i}"] = None


                        data_list.append(row)
                    except Exception as e:
                        print(f"Error {e} en el archivo: {file_path}")
                        traceback.print_exc()
                        print("-" * 60)
    
    df = pd.DataFrame(data_list)

    df.to_csv(output_csv, index=False)
    print(f"Procesos terminado, se han guardado f{len(df)} filas en {output_csv}")

# A partir de la ruta del json de anotaciones obtengo la ruta de la imagen
def _get_path(json_path): 
    
    img_path = ""
    json_file = os.path.basename(json_path)
    img_name = json_file.replace('.json', '.jpg')


    if re.search(r'train', json_path, re.IGNORECASE):
        img_path = os.path.join(AFFECT_NET_TRAIN_PATH, img_name)
    elif re.search(r'val', json_path, re.IGNORECASE):
        img_path = os.path.join(AFFECTNET_VAL_PATH, img_name)
    elif re.search(r'no_human_annotated', json_path, re.IGNORECASE):
        img_path = os.path.join(AFFECTNET_NO_HUMAN_ANNOTATED_PATH, img_name)
    else:
        print(f"Error en la ruta: {json_path}")
        return None
    
    return img_path

if __name__ == "__main__":

    print("Parseando archivos en train")
    output_csv = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/train.csv"
    migrate_json_to_csv(JSON_DIR_TRAIN, output_csv)

    print("Parseando archivos en val")
    output_csv = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/val.csv"
    migrate_json_to_csv(JSON_DIR_VAL, output_csv)

    print("Parseando archivos no_human_annotated")
    output_csv = "/home12TB1/database/recognition/faces/affectnet/img2pose_ann/no_human_annotated.csv"
    migrate_json_to_csv(JSON_DIR_NOHUMANANNOTATED, output_csv)
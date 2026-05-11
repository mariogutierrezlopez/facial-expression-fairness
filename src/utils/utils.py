# Utils.py
# Mario Gutiérrez López

import pandas as pd
from sklearn.model_selection import train_test_split

POSE_BINS = {
    "frontal": ["14_0", "05_1", "05_0"],
    "profile": ["12_0", "20_0", "01_0"],
}

def calc_nlimit_pose(csv_path, bias_arr: list):
    """
    Calcula el N_limit global para el experimento de sesgo de pose.
    Garantiza que haya suficientes imágenes para todas las clases y
    todos los factores 'g' en los 4 cruces de Género x Pose.
    """
    df = pd.read_csv(csv_path)
    df = generate_labels(df)

    def map_pose(cam):
        cam_str = str(cam)
        if cam_str in POSE_BINS["frontal"]: return "frontal"
        if cam_str in POSE_BINS["profile"]: return "profile"
        return "other"
        
    df['pose'] = df['camera_id'].apply(map_pose)
    df = df[df['pose'] != "other"]

    subjects_df = df[['subject_id', 'gender']].drop_duplicates()
    train_subs, _ = train_test_split(
        subjects_df,
        test_size=0.3,
        stratify=subjects_df['gender'],
        random_state=42
    )
    df = df[df['subject_id'].isin(train_subs['subject_id'])]
    
    classes = [c for c in df['temp_label'].unique() if c != -1]

    min_pose_limit = float('inf')

    for g in bias_arr:
        for label in classes:
            # Disponibilidad real en el dataset para esta clase
            avail_wf = len(df[(df['temp_label'] == label) & (df['gender'] == 'Female') & (df['pose'] == 'frontal')])
            avail_wp = len(df[(df['temp_label'] == label) & (df['gender'] == 'Female') & (df['pose'] == 'profile')])
            avail_mf = len(df[(df['temp_label'] == label) & (df['gender'] == 'Male') & (df['pose'] == 'frontal')])
            avail_mp = len(df[(df['temp_label'] == label) & (df['gender'] == 'Male') & (df['pose'] == 'profile')])

            f_gender = 0.5
            
            limits = []
            
            # Límites basados en frontales vs perfiles según 'g'
            if g > 0:
                limits.append(avail_wf / (f_gender * g))         # Mujeres frente
                limits.append(avail_mp / (f_gender * g))         # Hombres perfil
            
            if g < 1:
                limits.append(avail_wp / (f_gender * (1 - g)))   # Mujeres perfil
                limits.append(avail_mf / (f_gender * (1 - g)))   # Hombres frente
                
            # El límite máximo que soporta este escenario es el mínimo de los 4 grupos
            current_limit = int(min(limits))
            
            # Actualizamos el límite global si este es más restrictivo
            if current_limit < min_pose_limit:
                min_pose_limit = current_limit

    return int(min_pose_limit)

def calc_nlimits(csv_path, bias_arr:list):
    """
    Calcula 2 N_limit separados para maximizar los datos
    1. Representacional: Cuello de botella entre todos los factores f
    2. Estereotípico: Cuello de botella aun mayor

    :param df: Dataframe MultiPIE
    :param bias_arr: Array con combinaciones de sesgos [0.0, 0.25...]
    :type bias_arr: list
    """

    df = pd.read_csv(csv_path)
    df = generate_labels(df)

    # Calc n_limit for 70% train split
    subjects_df = df[['subject_id', 'gender']].drop_duplicates()
    train_subs, _ = train_test_split(
        subjects_df,
        test_size=0.3,
        stratify=subjects_df['gender'],
        random_state=42
    )

    df = df[df['subject_id'].isin(train_subs['subject_id'])]
    
    classes = [c for c in df['temp_label'].unique() if c != -1]

    min_repres = float('inf')

    for f in bias_arr:
        
        # EVALUAR SESGO REPRESENTACIONAL
        current_f_limit = _get_limit_for_factor(df, classes, f)
        if current_f_limit < min_repres:
            min_repres = current_f_limit

    # EVALUAR SESGO ESTEREOTÍPICO
    all_stereo_scenarios = []
    for f in bias_arr:
        for target_class in classes:
            scenario_limits = []
            for label in classes:
                current_f = f if label == target_class else 0.5

                n_w = len(df[(df['temp_label'] == label) & (df['gender'] == 'Female')])
                n_m = len(df[(df['temp_label'] == label) & (df['gender'] == 'Male')])
                
                limit_w = int(n_w / current_f) if current_f > 0 else float('inf')
                limit_m = int(n_m / (1 - current_f)) if current_f < 1 else float('inf')
                scenario_limits.append(min(limit_w, limit_m))

            all_stereo_scenarios.append(min(scenario_limits))
    min_stereo = min(all_stereo_scenarios)

    return int(min_repres), int(min_stereo)

def _get_limit_for_factor(df, classes, f):
    """Función auxiliar para calcular el límite de una configuración f específica"""
    limits = []
    for label in classes:
        n_w = len(df[(df['temp_label'] == label) & (df['gender'] == 'Female')])
        n_m = len(df[(df['temp_label'] == label) & (df['gender'] == 'Male')])

        limit_w = int(n_w / f) if f > 0 else float('inf')
        limit_m = int(n_m / (1 - f)) if f < 1 else float('inf')
        limits.append(min(limit_w, limit_m))
    return min(limits)

def generate_labels(df):
    """
    Generates the 'temp_label' column based on session and recording IDs.
    Returns the modified DataFrame.
    """
    def map_row(row):
        # Convert to string once to ensure matching
        session = str(row['session_id'])
        rec = int(row['recording_id'])

        if 'session01' in session:
            if rec == 1: return 0 # Neutral
            if rec == 2: return 1 # Smile
        if 'session02' in session:
            if rec == 1: return 0 # Neutral
            if rec == 2: return 2 # Surprise
            if rec == 3: return 3 # Squint
        if 'session03' in session:
            if rec == 1: return 0 # Neutral
            if rec == 2: return 1 # Smile
            if rec == 3: return 4 # Disgust
        if 'session04' in session:
            if rec == 1: return 0 # Neutral
            if rec == 2: return 0 # Neutral
            if rec == 3: return 5 # Scream
            
        return -1

    # Create the column and return the full DF
    df['temp_label'] = df.apply(map_row, axis=1)
    
    df = df[df['temp_label'] != -1].reset_index(drop=True)
    
    return df
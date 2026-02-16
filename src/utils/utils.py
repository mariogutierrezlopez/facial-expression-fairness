# Utils.py
# Mario Gutiérrez López

import pandas as pd

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
    classes = [c for c in df['temp_label'].unique() if c != -1]

    min_repres = float('inf')

    for f in bias_arr:
        
        # EVALUAR SESGO REPRESENTACIONAL
        current_f_limit = _get_limit_for_factor(df, classes, f)
        if current_f_limit < min_repres:
            min_repres = current_f_limit

        # EVALUAR SESGO ESTEREOTÍPICO
    limit_at_05 = _get_limit_for_factor(df, classes, 0.5)
    min_stereo = min(min_repres, limit_at_05)
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
        rec = row['recording_id']

        if rec == 1: 
            return 0  # Neutral
        
        # Session-specific logic
        if 'session01' in session:
            if rec == 2: return 1  # Smile
        elif 'session02' in session:
            if rec == 2: return 2  # Surprise
            if rec == 3: return 3  # Squint
        elif 'session03' in session:
            if rec == 2: return 1  # Smile
            if rec == 3: return 4  # Disgust
        elif 'session04' in session:
            if rec == 2: return 5  # Scream
            if rec == 3: return 0  # Neutral
            
        return -1

    # Create the column and return the full DF
    df['temp_label'] = df.apply(map_row, axis=1)
    
    df = df[df['temp_label'] != -1].reset_index(drop=True)
    
    return df
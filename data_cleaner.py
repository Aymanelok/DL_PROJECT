import pandas as pd
import re

def clean_tweet_text(text):
    """
    Fonction pour nettoyer un tweet brut.
    """
    # Sécurité supplémentaire au cas où un float/null serait passé
    if not isinstance(text, str):
        return ""

    # 1. Suppression des URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Suppression des mentions (@utilisateur)
    text = re.sub(r'\@\w+', '', text)
    
    # 3. Suppression du symbole Hashtag (on garde le mot, mais on enlève le '#')
    text = re.sub(r'\#', '', text)
    
    # 4. Suppression de la ponctuation et des caractères spéciaux
    # On ne garde que les lettres et les espaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # 5. Conversion en minuscules
    text = text.lower()
    
    # 6. Suppression des espaces multiples restants
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_dataset(file_path):
    """
    Charge le dataset, élimine les lignes nulles et applique le nettoyage.
    """
    print(f"Chargement du fichier : {file_path}")
    
    # Lecture du fichier parquet
    df = pd.read_parquet(file_path)
    
    # ÉTAPE CRUCIALE : Élimination des lignes nulles sur les colonnes importantes
    # 'text' est le tweet, 'HS' est le label de haine (Hate Speech)
    df = df.dropna(subset=['text', 'HS'])
    
    # Application de la fonction de nettoyage
    df['clean_text'] = df['text'].apply(clean_tweet_text)
    
    print(f"Nettoyage terminé. Lignes restantes : {len(df)}")
    return df

# --- Exemple d'utilisation ---
# df_train = prepare_dataset('dev-00000-of-00001.parquet')
# print(df_train[['text', 'clean_text']].head())

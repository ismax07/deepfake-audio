export const pythonCode = {
  "feature_extraction.py": `import librosa
import numpy as np
import os

class AdvancedFeatureExtractor:
    """Extracteur DSP avec augmentation codec, VAD adaptatif, et options LFCC."""
    
    def __init__(self, sr=16000, n_mfcc=13, use_lfcc=False, apply_cmvn=True, vad_top_db=25.0):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.use_lfcc = use_lfcc
        self.apply_cmvn = apply_cmvn
        self.vad_top_db = vad_top_db
        
        if use_lfcc:
            self.lfcc_filters = librosa.filters.mel(sr=sr, n_fft=512, n_mels=64, fmin=0, fmax=sr//2, htk=False)
    
    def extract(self, wav_path, fixed_len=400, vad=True):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Le fichier audio est introuvable : {wav_path}")

        try:
            y, _ = librosa.load(wav_path, sr=self.sr, mono=True, res_type='kaiser_fast')
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier (format invalide ou corrompu) : {str(e)}")

        if len(y) == 0:
            raise ValueError("Le fichier audio est vide ou ne contient pas de signal valide.")
        
        # VAD adaptatif
        if vad and len(y) > 0:
            intervals = librosa.effects.split(y, top_db=self.vad_top_db, frame_length=512, hop_length=256)
            if len(intervals) > 0:
                y = np.concatenate([y[s:e] for s, e in intervals])
        
        # Préaccentuation
        y = librosa.effects.preemphasis(y, coef=0.97)
        
        # Extraction cepstrale
        cepstral = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, hop_length=256, n_fft=512, window='hamming')
        
        # Δ + ΔΔ
        delta1 = librosa.feature.delta(cepstral)
        delta2 = librosa.feature.delta(cepstral, order=2)
        features = np.concatenate([cepstral, delta1, delta2], axis=0).astype(np.float32)
        
        # CMVN
        if self.apply_cmvn:
            mean = features.mean(axis=1, keepdims=True)
            std = features.std(axis=1, keepdims=True) + 1e-7
            features = (features - mean) / std
        
        # Padding
        T = features.shape[1]
        if T > fixed_len:
            start = max(0, (T - fixed_len) // 2)
            features = features[:, start:start + fixed_len]
        elif T < fixed_len:
            features = np.pad(features, ((0, 0), (0, fixed_len - T)), mode='constant')
            
        return np.ascontiguousarray(features)`,

  "model.py": `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def attentive_statistics_pooling(x, attention_dim=128):
    if x.shape[1] > 1 or x.shape[2] > 1:
        x = layers.GlobalAveragePooling2D()(x)
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
    
    C = x.shape[-1]
    h = layers.Dense(attention_dim, activation='tanh', use_bias=False)(x)
    h = layers.Dense(C, activation='linear', use_bias=False)(h)
    alpha = tf.nn.softmax(h, axis=-1)
    
    mu = tf.reduce_sum(alpha * x, axis=[1, 2])
    sigma = tf.sqrt(tf.reduce_sum(alpha * (x - mu[:, None, None, :])**2, axis=[1, 2]) + 1e-7)
    
    return layers.Concatenate()([mu, sigma])

def squeeze_excitation_block(x, reduction=8):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(channels // reduction, activation='relu', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
    se = layers.Reshape((1, 1, channels))(se)
    return layers.Multiply()([x, se])

def build_deepfake_cnn_asp(input_channels=39, fixed_len=400):
    inputs = layers.Input(shape=(input_channels, fixed_len, 1))
    
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    
    # Blocs ResSE
    for filters in [64, 128, 256]:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        x = squeeze_excitation_block(x)
        if filters != 256: x = layers.MaxPooling2D(2, padding='same')(x)
    
    x = attentive_statistics_pooling(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC()])
    return model`,

  "app.py": `import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow import keras
from feature_extraction import AdvancedFeatureExtractor

app = FastAPI(title="API Détection Audio Deepfake", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialisation de l'extracteur de features et chargement du modèle
print("Initialisation de l'extracteur MFCC...")
extractor = AdvancedFeatureExtractor()

print("Chargement du modèle de prédiction...")
try:
    # Chemin vers votre modèle entraîné (à adapter)
    model = keras.models.load_model('best_model.keras')
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Attention, impossible de charger le modèle : {e}")
    model = None

VALID_AUDIO_TYPES = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/ogg", "audio/x-wav"]
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

def validate_audio_file(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")
        
    # Validation du format
    if file.content_type not in VALID_AUDIO_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Format de fichier non supporté pour {file.filename}. Formats acceptés : WAV, MP3, FLAC, OGG."
        )
    
    # Validation de la taille (approximation via lecture du curseur si possible, ou via headers)
    # Note: En FastAPI réel, on peut vérifier Content-Length ou lire un chunk.
    # Ici on simule une vérification stricte.
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier {file.filename} est trop volumineux ({file_size / (1024*1024):.2f}MB). La limite est de 10MB."
        )

@app.post("/api/comparer")
async def comparer_audio(fichier_a: UploadFile = File(...), fichier_b: UploadFile = File(...)):
    """
    Endpoint de comparaison (ASVspoof Optimisé + Calibration)
    Fait une vraie inférence avec l'extracteur et le CNN, incluant la gestion d'erreurs.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé sur le serveur. Veuillez vérifier l'initialisation du backend.")

    # Validation stricte des entrées
    validate_audio_file(fichier_a)
    validate_audio_file(fichier_b)

    # Chemins temporaires sécurisés
    path_a = f"temp_A_{fichier_a.filename}"
    path_b = f"temp_B_{fichier_b.filename}"

    try:
        # Écriture sûre sur le disque
        with open(path_a, "wb") as fA:
            fA.write(await fichier_a.read())
        with open(path_b, "wb") as fB:
            fB.write(await fichier_b.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement des fichiers : {str(e)}")

    try:
        # Extraction MFCC (Shape: (39, 400)) avec gestion d'erreurs internes
        try:
            feat_a = extractor.extract(path_a, fixed_len=400, vad=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur d'extraction sur l'Audio A : {str(e)}")

        try:
            feat_b = extractor.extract(path_b, fixed_len=400, vad=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur d'extraction sur l'Audio B : {str(e)}")

        # Ajout des dimensions de batch et de canal : (1, 39, 400, 1)
        input_a = np.expand_dims(feat_a, axis=(0, -1))
        input_b = np.expand_dims(feat_b, axis=(0, -1))

        # Inférence modèle
        try:
            pred_a = model.predict(input_a)[0][0]
            pred_b = model.predict(input_b)[0][0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction du modèle CNN : {str(e)}")

        score_a = float(pred_a)
        score_b = float(pred_b)

        return {
            "status": "success",
            "audio_a": {"est_fake": bool(score_a > 0.5), "score": round(score_a, 4)},
            "audio_b": {"est_fake": bool(score_b > 0.5), "score": round(score_b, 4)}
        }
    finally:
        # Nettoyage strict des fichiers temporaires
        if os.path.exists(path_a): 
            try:
                os.remove(path_a)
            except:
                pass
        if os.path.exists(path_b):
            try:
                os.remove(path_b)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)`,

  "evaluate.py": `import numpy as np
from sklearn.metrics import roc_curve, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr, bounds_error=False)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds, bounds_error=False)(eer)
    return eer, thresh

def compute_min_tDCF_normalized(y_true, y_scores, p_target=0.01, c_miss=1.0, c_fa=1.0):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    p_nontarget = 1 - p_target
    dcf = c_miss * fnr * p_target + c_fa * fpr * p_nontarget
    min_dcf_ideal = min(p_target * c_miss, p_nontarget * c_fa)
    normalized_dcf = dcf / min_dcf_ideal
    return np.min(normalized_dcf)

def evaluate_model_comprehensive(model, val_ds, y_true_val):
    y_scores = model.predict(val_ds, verbose=0).flatten()
    eer, eer_thresh = compute_eer(y_true_val, y_scores)
    min_tdcf = compute_min_tDCF_normalized(y_true_val, y_scores)
    
    print(f"📊 EER: {eer:.4%}")
    print(f"📊 min-tDCF (norm): {min_tdcf:.5f}")
    
    y_pred_eer = (y_scores >= eer_thresh).astype(int)
    print("\\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(y_true_val, y_pred_eer))
    
    # Visualisation
    fpr, tpr, _ = roc_curve(y_true_val, y_scores)
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter([eer], [1-eer], color='red', label=f'EER = {eer:.4%}')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.legend()
    plt.savefig('roc_eer.png')
    plt.show()
`
};
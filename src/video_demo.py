import cv2
from ultralytics import YOLO
import time
import os

# --- CONFIGURATION ---
# Chemins des fichiers
MODEL_PATH = 'models/drowsy_v2.pt'
INPUT_VIDEO = 'data/test.mp4'      # vidéo source 
OUTPUT_VIDEO = 'demo_result.avi'    # La vidéo qui sera créée

# Tes 11 classes exactes
CLASS_NAMES = [
    'Attentive eye', 'Drowsy eye', 'Eyeclosed', 'Open-Mouth', 
    'Yawn', 'asleep', 'close', 'closed', 'noYawn', 'open', 'yawn'
]

# Classes dangereuses
DANGER_CLASSES = [
    'Drowsy eye', 'Eyeclosed', 'Yawn', 'asleep', 
    'close', 'closed', 'yawn'
]

ALARM_TRIGGER_TIME = 2.0
CONFIDENCE_THRESHOLD = 0.40

# --- INITIALISATION ---
print(f"Chargement du modèle : {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Erreur modèle : {e}")
    exit()

# Vérifier si la vidéo d'entrée existe
if not os.path.exists(INPUT_VIDEO):
    print(f"ERREUR : La vidéo '{INPUT_VIDEO}' n'existe pas à la racine du projet.")
    exit()

# Chargement de la vidéo
cap = cv2.VideoCapture(INPUT_VIDEO)

# Récupération des infos de la vidéo (pour créer la sortie)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Création de l'enregistreur vidéo (Codec XVID pour le format .avi)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

print(f"🎬 Traitement de la vidéo en cours... (Résultat dans {OUTPUT_VIDEO})")
start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Fin de la vidéo

    # --- MÊME LOGIQUE QUE INFERENCE.PY ---
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    driver_is_tired = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_NAMES[cls_id]

            if label in DANGER_CLASSES:
                color = (0, 0, 255)
                driver_is_tired = True
            elif label == 'Open-Mouth':
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if driver_is_tired:
        if start_time is None: start_time = time.time()
        elapsed = time.time() - start_time
        cv2.putText(frame, f"FATIGUE: {elapsed:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed > ALARM_TRIGGER_TIME:
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), 10)
            cv2.putText(frame, "!!! ALERTE !!!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            # Pas de winsound ici, ça ne sert à rien dans une vidéo enregistrée
    else:
        start_time = None
        cv2.putText(frame, "VIGILANT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- ENREGISTREMENT ---
    # On écrit l'image modifiée dans le fichier de sortie
    out.write(frame)
    
    # Optionnel : afficher pendant le traitement (ralentit le processus)
    cv2.imshow('Traitement Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release() # Très important de relâcher le fichier de sortie
cv2.destroyAllWindows()
print("Traitement terminé ! Vidéo enregistrée.")
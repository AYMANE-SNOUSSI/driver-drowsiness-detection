import cv2
from ultralytics import YOLO
import time
import winsound # Pour le bip sonore sur Windows

# --- CONFIGURATION ---
# 1. Chemin du modèle
MODEL_PATH = 'models/drowsy_v2.pt' 

# 2. 11 classes exactes (Ne change pas l'ordre)
CLASS_NAMES = [
    'Attentive eye', # 0 - OK
    'Drowsy eye',    # 1 - DANGER
    'Eyeclosed',     # 2 - DANGER
    'Open-Mouth',    # 3 - ATTENTION (Bâillement ou parole ?)
    'Yawn',          # 4 - DANGER
    'asleep',        # 5 - DANGER EXTREME
    'close',         # 6 - DANGER
    'closed',        # 7 - DANGER
    'noYawn',        # 8 - OK
    'open',          # 9 - OK
    'yawn'           # 10 - DANGER
]

# 3. On liste les mots-clés qui déclenchent l'alarme
DANGER_CLASSES = [
    'Drowsy eye', 'Eyeclosed', 'Yawn', 'asleep', 
    'close', 'closed', 'yawn'
]

ALARM_TRIGGER_TIME = 2.0  # Alarme après 2 secondes
CONFIDENCE_THRESHOLD = 0.40 # Seuil de confiance

# --- INITIALISATION ---
print(f"Chargement du modèle : {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except:
    print(f"ERREUR : Le fichier {MODEL_PATH} est introuvable.")
    print("Vérifie que tu l'as bien mis dans le dossier 'models' !")
    exit()

# Connexion à OBS (Index 2)
print("Connexion à la caméra (OBS)...")
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERREUR CAMÉRA : Vérifie que OBS Virtual Camera est démarrée.")
    # Si OBS ne marche pas, essaye cap = cv2.VideoCapture(0)
    exit()

start_time = None

print("Système prêt ! Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Détection YOLO
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    
    driver_is_tired = False
    
    # Analyse des résultats
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Nom de la classe détectée
            label = CLASS_NAMES[cls_id]

            # Est-ce dangereux ?
            if label in DANGER_CLASSES:
                color = (0, 0, 255) # Rouge
                driver_is_tired = True
            elif label == 'Open-Mouth':
                color = (0, 165, 255) # Orange
            else:
                color = (0, 255, 0) # Vert

            # Dessiner la boîte
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- LOGIQUE D'ALERTE ---
    if driver_is_tired:
        if start_time is None:
            start_time = time.time()
        
        elapsed = time.time() - start_time
        
        # Affichage Timer Rouge
        cv2.putText(frame, f"FATIGUE: {elapsed:.1f}s", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed > ALARM_TRIGGER_TIME:
            # ALERTE MAXIMALE
            cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 255), 10)
            cv2.putText(frame, "!!! REVEIL !!!", (150, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            try:
                winsound.Beep(2000, 100) # Biiiiip
            except: pass
    else:
        start_time = None
        cv2.putText(frame, "VIGILANT", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detection Fatigue V2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
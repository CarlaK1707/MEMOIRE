import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import cv2
from ultralytics import YOLOv10
import math
import easyocr
import numpy as np
import re
from datetime import datetime
import json
import sqlite3
import pandas as pd
import logging
from collections import defaultdict, Counter

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialiser EasyOCR
reader = easyocr.Reader(['en'], model_storage_directory='./model_cache', download_enabled=True, user_network_directory='./custom_model')

# Créer l'objet Video Capture
cap = cv2.VideoCapture("testF.mp4")

# Obtenir les propriétés vidéo
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir le codec et créer VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output9.mp4', fourcc, fps, (width, height))

# Initialiser YOLOv10 Model
model = YOLOv10("weights/best.pt")
count = 0
className = ['Immatriculation']
print("RECONNAISSANCE DE PLAQUE")

# Charger la base de données Excel avec plaques et noms
authorized_data = {}
try:
    df_plates = pd.read_excel('authorized_plates.xlsx')
    logger.info(f"Colonnes disponibles dans Excel : {df_plates.columns.tolist()}")
    plate_column = 'licence_plate' if 'licence_plate' in df_plates.columns else df_plates.columns[1]
    name_column = 'name' if 'name' in df_plates.columns else None
    if name_column:
        authorized_data = {str(row[plate_column]).upper().strip(): row[name_column] for index, row in df_plates.iterrows() if pd.notna(row[plate_column])}
    else:
        logger.warning("Colonne 'name' non trouvée. Utilisation uniquement des plaques.")
        authorized_data = {str(plate).upper().strip(): None for plate in df_plates[plate_column].dropna().astype(str).str.strip()}
    logger.info("Base de données Excel chargée avec succès")
except FileNotFoundError:
    logger.error("Fichier Excel 'authorized_plates.xlsx' non trouvé. Vérifiez le chemin.")
    exit(1)
except Exception as e:
    logger.error(f"Erreur lors du chargement de la base de données Excel : {e}")
    exit(1)

# Système de suivi des plaques par véhicule
class PlateTracker:
    def __init__(self):
        self.vehicle_plates = {}  # {vehicle_id: {plate: count}}
        self.stable_plates = {}   # {vehicle_id: confirmed_plate}
        self.confirmation_threshold = 5  # Nombre de détections nécessaires pour confirmer
        self.max_history = 20    # Historique maximum par véhicule
        
    def get_vehicle_id(self, x1, y1, x2, y2):
        """Générer un ID de véhicule basé sur la position (simplifié)"""
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # Grouper par zones de 100x100 pixels
        zone_x = center_x // 100
        zone_y = center_y // 100
        return f"{zone_x}_{zone_y}"
    
    def add_detection(self, vehicle_id, plate_text):
        """Ajouter une détection de plaque pour un véhicule"""
        if vehicle_id not in self.vehicle_plates:
            self.vehicle_plates[vehicle_id] = Counter()
        
        if plate_text:
            self.vehicle_plates[vehicle_id][plate_text] += 1
            
            # Limiter l'historique
            if len(self.vehicle_plates[vehicle_id]) > self.max_history:
                # Garder seulement les plus fréquentes
                most_common = dict(self.vehicle_plates[vehicle_id].most_common(self.max_history // 2))
                self.vehicle_plates[vehicle_id] = Counter(most_common)
    
    def get_stable_plate(self, vehicle_id):
        """Obtenir la plaque stable pour un véhicule"""
        if vehicle_id in self.stable_plates:
            return self.stable_plates[vehicle_id]
        
        if vehicle_id in self.vehicle_plates:
            # Trouver la plaque la plus fréquente
            most_common = self.vehicle_plates[vehicle_id].most_common(1)
            if most_common and most_common[0][1] >= self.confirmation_threshold:
                confirmed_plate = most_common[0][0]
                self.stable_plates[vehicle_id] = confirmed_plate
                return confirmed_plate
                
        return None
    
    def cleanup_old_vehicles(self):
        """Nettoyer les anciens véhicules (appelé périodiquement)"""
        # Ici vous pourriez implémenter une logique plus sophistiquée
        # basée sur le temps ou la position
        pass

# Initialiser le tracker
plate_tracker = PlateTracker()

def preprocess_image(roi):
    """Prétraitement optimisé pour la reconnaissance de plaques"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Redimensionner pour une meilleure reconnaissance
    height, width = gray.shape
    scale_factor = 3 if min(width, height) < 100 else 2
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    
    # Débruitage
    denoised = cv2.medianBlur(enhanced, 3)
    
    # Accentuation des contours
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def validate_license_plate(text):
    """Validation améliorée pour différents formats de plaques"""
    if not text:
        return False
    
    # Nettoyer le texte
    cleaned = re.sub(r'[^0-9A-Z]', '', text.upper().strip())
    
    # Longueur minimale et maximale
    if len(cleaned) < 5 or len(cleaned) > 12:
        return False
    
    # Formats acceptés (adaptez selon votre région)
    patterns = [
        r'^[0-9]{4}[A-Z]{2}[0-9]{2}$',  # Format: 1234AB56
        r'^[0-9]{4}[A-Z]{2}[0-9]{1}$',  # Format: 1234AB5
        r'^[0-9]{3}[A-Z]{2}[0-9]{2}$',  # Format: 123AB56
        r'^[0-9]{4}[A-Z]{3}[0-9]{2}$', # Format: 1234ABC56
        r'^[A-Z]{2}[0-9]{3}[A-Z]{2}$',  # Format: AB123CD
        r'^[0-9]{3}[A-Z]{3}[0-9]{2}$', # Format: 123ABC56
    ]
    
    for pattern in patterns:
        if re.match(pattern, cleaned):
            return cleaned
    
    # Validation flexible : au moins 2 lettres et 2 chiffres
    letter_count = len(re.findall(r'[A-Z]', cleaned))
    digit_count = len(re.findall(r'[0-9]', cleaned))
    
    if letter_count >= 2 and digit_count >= 2:
        return cleaned
    
    return False

def similarity_score(text1, text2):
    """Calculer la similarité entre deux textes"""
    if not text1 or not text2:
        return 0
    
    # Longueur similaire
    len_diff = abs(len(text1) - len(text2))
    if len_diff > 2:
        return 0
    
    # Caractères en commun
    common_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
    max_len = max(len(text1), len(text2))
    
    return common_chars / max_len if max_len > 0 else 0

def find_closest_authorized_plate(detected_plate):
    """Trouver la plaque autorisée la plus proche"""
    if not detected_plate:
        return None, 0
    
    best_match = None
    best_score = 0
    
    for auth_plate in authorized_data.keys():
        score = similarity_score(detected_plate, auth_plate)
        if score > best_score and score > 0.8:  # Seuil de similarité
            best_score = score
            best_match = auth_plate
    
    return best_match, best_score

def paddle_ocr(frame, x1, y1, x2, y2):
    """Fonction OCR améliorée avec correction automatique"""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return ""
    
    # Prétraitement optimisé
    preprocessed = preprocess_image(roi)
    
    detected_texts = []
    
    # Tentative 1: Image prétraitée standard
    try:
        result1 = reader.readtext(preprocessed, detail=1, paragraph=False, 
                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                 width_ths=0.8, height_ths=0.8)
        
        for detection in result1:
            text = detection[1].upper().strip()
            confidence = detection[2]
            if confidence > 0.4:
                detected_texts.append((text, confidence))
    except:
        pass
    
    # Tentative 2: Avec ajustement de contraste
    try:
        contrast_enhanced = cv2.convertScaleAbs(preprocessed, alpha=1.8, beta=20)
        result2 = reader.readtext(contrast_enhanced, detail=1, paragraph=False,
                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                 width_ths=0.6, height_ths=0.6)
        
        for detection in result2:
            text = detection[1].upper().strip()
            confidence = detection[2]
            if confidence > 0.3:
                detected_texts.append((text, confidence))
    except:
        pass
    
    # Tentative 3: Binarisation
    try:
        binary = cv2.adaptiveThreshold(preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, 2)
        result3 = reader.readtext(binary, detail=1, paragraph=False,
                                 allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                 width_ths=0.5, height_ths=0.5)
        
        for detection in result3:
            text = detection[1].upper().strip()
            confidence = detection[2]
            if confidence > 0.25:
                detected_texts.append((text, confidence))
    except:
        pass
    
    # Analyser tous les textes détectés
    best_validated = None
    best_confidence = 0
    
    for text, confidence in detected_texts:
        validated = validate_license_plate(text)
        if validated:
            # Vérifier s'il y a une correspondance proche dans la base
            closest_match, match_score = find_closest_authorized_plate(validated)
            
            # Bonus de confiance si correspond à une plaque autorisée
            final_confidence = confidence
            if closest_match and match_score > 0.9:
                final_confidence *= 1.5  # Bonus pour correspondance exacte
                validated = closest_match  # Utiliser la plaque correcte
            elif closest_match and match_score > 0.8:
                final_confidence *= 1.2  # Bonus modéré
                validated = closest_match  # Corriger vers la plaque autorisée
            
            if final_confidence > best_confidence:
                best_validated = validated
                best_confidence = final_confidence
    
    return best_validated if best_confidence > 0.3 else ""

def save_json(licence_plates, startTime, endTime):
    """Sauvegarde JSON inchangée"""
    interval_data = {
        "Heure de début": startTime.isoformat(),
        "Heure de fin": endTime.isoformat(),
        "Plaques": [{"plaque": plate, "nom": authorized_data.get(plate.upper(), "Inconnu"), 
                    "statut_acces": 'ACCES AUTORISE' if plate.upper() in authorized_data else 'ACCES NON AUTORISE'} 
                   for plate in licence_plates]
    }
    os.makedirs("json", exist_ok=True)
    interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(interval_file_path, 'w', encoding='utf-8') as f:
        json.dump(interval_data, f, indent=2, ensure_ascii=False)

    cummulative_file_path = "json/LicencePlateData.json"
    existing_data = []
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    existing_data.append(interval_data)
    with open(cummulative_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    save_to_database(licence_plates, startTime, endTime)

def save_to_database(licence_plates, start_time, end_time):
    """Sauvegarde base de données inchangée"""
    conn = sqlite3.connect('licencePlates.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS LicencesPlates (start_time TEXT, end_time TEXT, licence_plate TEXT, name TEXT, access_status TEXT)")
    for plate in licence_plates:
        name = authorized_data.get(plate.upper(), "Inconnu")
        access_status = 'ACCES AUTORISE' if plate.upper() in authorized_data else 'ACCES NON AUTORISE'
        cursor.execute('INSERT INTO LicencesPlates (start_time, end_time, licence_plate, name, access_status) VALUES (?, ?, ?, ?, ?)',
                       (start_time.isoformat(), end_time.isoformat(), plate, name, access_status))
    conn.commit()
    conn.close()

# Boucle principale
startTime = datetime.now()
license_plates = set()
frame_cache = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    currentTime = datetime.now()
    count += 1
    print(f"Numéro de frame : {count}")
    
    if frame is not None and str(frame.tobytes()) not in frame_cache:
        results = model.predict(frame, conf=0.3)  # Seuil encore plus bas
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Agrandir la boîte de détection
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                # Obtenir l'ID du véhicule
                vehicle_id = plate_tracker.get_vehicle_id(x1, y1, x2, y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Reconnaissance OCR
                detected_plate = paddle_ocr(frame, x1, y1, x2, y2)
                
                # Ajouter au tracker
                plate_tracker.add_detection(vehicle_id, detected_plate)
                
                # Obtenir la plaque stable
                stable_plate = plate_tracker.get_stable_plate(vehicle_id)
                
                # Utiliser la plaque stable ou la détection actuelle
                final_plate = stable_plate if stable_plate else detected_plate
                
                if final_plate:
                    license_plates.add(final_plate)
                    access_status = 'ACCES AUTORISE' if final_plate.upper() in authorized_data else 'ACCES NON AUTORISE'
                    name = authorized_data.get(final_plate.upper(), "Inconnu")
                    
                    # Couleur selon le statut et la stabilité
                    if stable_plate:
                        color = (0, 255, 0) if access_status == 'ACCES AUTORISE' else (0, 0, 255)
                        status_prefix = "STABLE - "
                    else:
                        color = (0, 255, 255) if access_status == 'ACCES AUTORISE' else (0, 100, 255)
                        status_prefix = "DETECT - "
                    
                    # Affichage amélioré
                    detection_count = plate_tracker.vehicle_plates[vehicle_id][final_plate] if vehicle_id in plate_tracker.vehicle_plates else 0
                    text_display = f"Plaque: {final_plate} ({detection_count})\nStatut: {status_prefix}{access_status}\nNom: {name}"
                    lines = text_display.split('\n')
                    
                    # Fond semi-transparent
                    max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0] for line in lines])
                    bg_height = len(lines) * 30 + 10
                    
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - bg_height - 10), (x1 + max_width + 20, y1), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                    
                    # Texte
                    for i, line in enumerate(lines):
                        y_offset = y1 - bg_height + (i * 30) + 25
                        cv2.putText(frame, line, (x1 + 10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                else:
                    # Affichage si aucune plaque détectée
                    cv2.putText(frame, "Analyse en cours...", (x1 + 5, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        frame_cache[str(frame.tobytes())] = True
    
    out.write(frame)
    
    try:
        cv2.imshow("Vidéo originale", frame)
    except cv2.error:
        print("Avertissement : cv2.imshow non pris en charge. Sauvegarde dans la vidéo.")
    
    # Nettoyage périodique
    if count % 100 == 0:
        plate_tracker.cleanup_old_vehicles()
    
    # Sauvegarde périodique
    if (currentTime - startTime).seconds >= 20:
        endTime = currentTime
        save_json(license_plates, startTime, endTime)
        startTime = currentTime
        license_plates.clear()
        frame_cache.clear()
    
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
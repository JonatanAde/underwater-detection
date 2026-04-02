import cv2
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# --- FUNZIONE PER CREARE UN NUOVO FILTRO PER OGNI PESCE ---
def create_ukf(start_x, start_y):
    def transition_func(x, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        return F @ x
    def measurement_func(x):
        return np.array([x[0], x[1]])
    
    sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
    new_ukf = UKF(dim_x=4, dim_z=2, fx=transition_func, hx=measurement_func, dt=0.033, points=sigmas)
    new_ukf.x = np.array([start_x, start_y, 0, 0])
    new_ukf.R = np.eye(2) * 5
    new_ukf.Q = np.eye(4) * 0.1
    return new_ukf

# --- ELABORAZIONE VIDEO ---
cap = cv2.VideoCapture('fish.mp4')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('multi_tracking_pesci.mp4', fourcc, 30, (prev_frame.shape[1], prev_frame.shape[0]))

trackers = [] # Lista che conterrà i nostri filtri UKF

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, event_sim = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    prev_gray = gray.copy()
    display = cv2.cvtColor(event_sim, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(event_sim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_measurements = []
    for cnt in contours:
        if 100 < cv2.contourArea(cnt) < 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w/2, y + h/2
            current_measurements.append({'pos': np.array([cx, cy]), 'rect': (x, y, w, h)})

    # AGGIORNAMENTO FILTRI ESISTENTI
    assigned_measurements = set()
    for ukf in trackers:
        ukf.predict()
        # Cerchiamo la misura più vicina per questo tracker
        best_dist = 60 # Raggio massimo di aggancio
        best_idx = -1
        for i, m in enumerate(current_measurements):
            dist = np.linalg.norm(m['pos'] - ukf.x[:2])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx != -1:
            ukf.update(current_measurements[best_idx]['pos'])
            assigned_measurements.add(best_idx)
            # Disegno
            ex, ey = int(ukf.x[0]), int(ukf.x[1])
            cv2.circle(display, (ex, ey), 8, (0, 255, 0), 2) # UKF Verde
            x, y, w, h = current_measurements[best_idx]['rect']
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 1) # Box Rosso

    # CREAZIONE NUOVI TRACKER PER MISURE NON ASSEGNATE
    for i, m in enumerate(current_measurements):
        if i not in assigned_measurements:
            trackers.append(create_ukf(m['pos'][0], m['pos'][1]))

    # Pulizia: rimuovi tracker che escono dal video o non trovano pesci (opzionale per 3 cfu)
    if len(trackers) > 15: trackers = trackers[-15:]

    out.write(display)

cap.release()
out.release()
print("Video Multi-Tracking pronto!")
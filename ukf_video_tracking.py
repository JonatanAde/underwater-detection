import cv2
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# --- CONFIGURAZIONE UKF ---

def transition_func(x, dt):
    """ Modello di transizione: x_nuovo = x_vecchio + v * dt """
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x

def measurement_func(x):
    """ Cosa vede la camera: solo la posizione (x, y) """
    return np.array([x[0], x[1]])

# Inizializziamo Sigma Points e il filtro
# alpha, beta, kappa controllano come i punti si distribuiscono intorno alla media
sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
ukf = UKF(dim_x=4, dim_z=2, fx=transition_func, hx=measurement_func, dt=0.1, points=sigmas)

# Matrice di incertezza iniziale
ukf.P *= 10 
# Rumore di processo (quanto può cambiare improvvisamente il movimento)
ukf.Q = np.eye(4) * 0.1
# Rumore di misura (quanto è "ballerina" la rilevazione di OpenCV)
ukf.R = np.eye(2) * 5

# --- LOOP VIDEO ---

cap = cv2.VideoCapture('fish_solo_eventi.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Rilevamento (Detection)
    mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroid = None
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = np.array([float(x + w//2), float(y + h//2)])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # Box Blu = Rilevamento
            break

    # 2. Filtro di Kalman (UKF)
    ukf.predict() # Predizione basata sulla fisica
    
    if centroid is not None:
        ukf.update(centroid) # Correzione basata sulla camera
        
    # Estraiamo la posizione filtrata
    ux, uy = ukf.x[0], ukf.x[1]

    # 3. Visualizzazione
    # Punto Rosso = Misura grezza, Cerchio Verde = UKF (pulito)
    if centroid is not None:
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
    cv2.circle(frame, (int(ux), int(uy)), 8, (0, 255, 0), 2)
    cv2.putText(frame, "UKF Tracking", (int(ux)-20, int(uy)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('UKF Underwater Thesis Prototype', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
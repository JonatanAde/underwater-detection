import cv2
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# =============================
# CONFIGURAZIONE
# =============================

video_path = "fish_solo_eventi.mp4"
dt = 1/30.0

# -----------------------------
# MODELLO COORDINATED TURN
# -----------------------------
def fx(x, dt):
    px, py, v, theta, omega = x

    if abs(omega) > 1e-4:
        px += (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta))
        py += (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta))
    else:
        px += v * dt * np.cos(theta)
        py += v * dt * np.sin(theta)

    theta += omega * dt
    return np.array([px, py, v, theta, omega])

def hx(x):
    return np.array([x[0], x[1]])

sigmas = MerweScaledSigmaPoints(5, alpha=0.1, beta=2., kappa=0.)
ukf = UKF(dim_x=5, dim_z=2, fx=fx, hx=hx, dt=dt, points=sigmas)

# Stato iniziale (verrà sovrascritto al primo aggancio)
ukf.x = np.array([0., 0., 40., 0., 0.])
ukf.P *= 300

# Rumori (pesci dinamici → Q alto)
ukf.Q = np.diag([0.2, 0.2, 8.0, 0.1, 0.5])
ukf.R = np.eye(2) * 25

initialized = False
missed_frames = 0
max_missed = 60   # NON spegniamo subito

# =============================
# VIDEO SETUP
# =============================

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Errore apertura video")
    exit()

height, width = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("ukf_single_fish_clean.mp4",
                      fourcc, 30, (width, height))

kernel = np.ones((3,3), np.uint8)

print("Tracking avviato...")

# =============================
# LOOP PRINCIPALE
# =============================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Video già event-like → soglia semplice
    _, mask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
    mask = cv2.dilate(mask, kernel, 1)

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if 150 < area < 4000:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                valid.append((cx, cy, c))

    measurement = None
    best_cnt = None

    # -------------------------
    # INIZIALIZZAZIONE
    # -------------------------
    if not initialized and len(valid) > 0:
        cx, cy, best_cnt = valid[0]
        ukf.x = np.array([cx, cy, 40., 0., 0.])
        initialized = True
        print("Target agganciato")

    # -------------------------
    # TRACKING
    # -------------------------
    if initialized:

        ukf.predict()

        # Costruzione S corretta (no ukf.S!)
        H = np.array([
            [1,0,0,0,0],
            [0,1,0,0,0]
        ])

        P = ukf.P
        S = H @ P @ H.T + ukf.R
        S += np.eye(2) * 1e-6  # stabilità numerica

        min_dist = float('inf')

        for cx, cy, c in valid:
            z = np.array([cx, cy])
            y = z - hx(ukf.x)

            try:
                d = y.T @ np.linalg.solve(S, y)
            except:
                continue

            # gating largo (ambiente caotico)
            if d < 25 and d < min_dist:
                min_dist = d
                measurement = z
                best_cnt = c

        if measurement is not None:
            ukf.update(measurement)
            missed_frames = 0
        else:
            missed_frames += 1

        if missed_frames > max_missed:
            initialized = False
            print("Target perso")

    # -------------------------
    # DISEGNO (NO TRAIETTORIA)
    # -------------------------

    display = frame.copy()

    if initialized:
        est_x = int(ukf.x[0])
        est_y = int(ukf.x[1])

        # Cerchio verde = stima UKF
        cv2.circle(display, (est_x, est_y), 10, (0,255,0), 2)

        # Misura rossa
        if measurement is not None:
            mx = int(measurement[0])
            my = int(measurement[1])
            cv2.circle(display, (mx, my), 4, (0,0,255), -1)

    out.write(display)

cap.release()
out.release()

print("Video salvato: ukf_single_fish_clean.mp4")
import cv2

import numpy as np

from filterpy.kalman import UnscentedKalmanFilter as UKF

from filterpy.kalman import MerweScaledSigmaPoints



# --- 1. MODELLO DINAMICO NON LINEARE (Coordinated Turn) ---

def transition_func(x, dt):

    # x = [px, py, v, theta, omega]

    px, py, v, theta, omega = x

    new_x = np.zeros_like(x)

   

    if abs(omega) < 0.001: # Movimento quasi rettilineo

        new_x[0] = px + v * np.cos(theta) * dt

        new_x[1] = py + v * np.sin(theta) * dt

    else: # Virata coordinata

        new_x[0] = px + (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta))

        new_x[1] = py + (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta))

   

    new_x[2] = v

    new_x[3] = theta + omega * dt

    new_x[4] = omega

    return new_x



def measurement_func(x):

    # Misuriamo solo la posizione x e y (primi due elementi dello stato)

    return np.array([x[0], x[1]])



# --- 2. CONFIGURAZIONE UKF (5 DIMENSIONI) ---

sigmas = MerweScaledSigmaPoints(5, alpha=.1, beta=2., kappa=0.)

ukf = UKF(dim_x=5, dim_z=2, fx=transition_func, hx=measurement_func, dt=0.033, points=sigmas)



# Matrici di Rumore aggiornate per 5 dimensioni

ukf.R = np.eye(2) * 5

ukf.Q = np.diag([1, 1, 2, 1, 0.5])

#ukf.Q = np.diag([0.1, 0.1, 1.5, 0.5, 0.2])

ukf.P *= 10 # Incertezza iniziale



initialized = False



# --- 3. ELABORAZIONE VIDEO ---

cap = cv2.VideoCapture('fish.mp4')

ret, prev_frame = cap.read()

if not ret: exit()



prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

height, width = prev_gray.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('tracking_pesci_ukf_non_lineare.mp4', fourcc, 30, (width, height))



while cap.isOpened():

    ret, frame = cap.read()

    if not ret: break



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray)

    _, event_sim = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    prev_gray = gray.copy()



    display = cv2.cvtColor(event_sim, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(event_sim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   

    raw_measurement = None

    best_cnt = None

   

    if contours:

        if not initialized:

            for cnt in contours:

                area = cv2.contourArea(cnt)

                if 80 < area < 2000:

                    M = cv2.moments(cnt)

                    if M["m00"] > 0:

                        cx = float(M["m10"] / M["m00"])

                        cy = float(M["m01"] / M["m00"])

                        # --- FIX: Inizializzazione a 5 dimensioni ---

                        ukf.x = np.array([cx, cy, 2.0, 0.0, 0.0])

                        initialized = True

                        break

        else:

            min_dist = float('inf')

            pred_x, pred_y = ukf.x[0], ukf.x[1]

            for cnt in contours:

                area = cv2.contourArea(cnt)

                if 80 < area < 2000:

                    M = cv2.moments(cnt)

                    if M["m00"] > 0:

                        cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]

                        dist = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)

                        if dist < min_dist and dist < 120:

                            min_dist = dist

                            raw_measurement = np.array([cx, cy])

                            best_cnt = cnt



    # --- 4. PREDIZIONE E AGGIORNAMENTO ---

    if initialized:

        # Protezione numerica per la stabilità dell'UKF

        ukf.P = (ukf.P + ukf.P.T) / 2 + np.eye(5) * 1e-6

        ukf.predict()

       

        if raw_measurement is not None:

            ukf.update(raw_measurement)



        # --- 5. DISEGNO ---

        # Misura Grezza (Rosso)

        if raw_measurement is not None:

            x_b, y_b, w_b, h_b = cv2.boundingRect(best_cnt)

            cv2.rectangle(display, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 2)

            cv2.circle(display, (int(raw_measurement[0]), int(raw_measurement[1])), 3, (0, 0, 255), -1)

       

        # Stima UKF (Verde)

        ex, ey = int(ukf.x[0]), int(ukf.x[1])

        cv2.circle(display, (ex, ey), 10, (0, 255, 0), 2)

       

        # Vettore di direzione (per far vedere che l'angolo theta cambia)

        vx = int(ex + np.cos(ukf.x[3]) * 20)

        vy = int(ey + np.sin(ukf.x[3]) * 20)

        cv2.line(display, (ex, ey), (vx, vy), (0, 255, 0), 2)



        cv2.putText(display, "Misura Grezza", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "Stima UKF", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(display)



cap.release()

out.release()

print("Tracking non lineare completato!")
import numpy as np

def transition_function(state, dt):
    """
    Modello CTRV: Prevede la nuova posizione basandosi su 
    velocità lineare (v) e velocità angolare (omega).
    """
    x, y, v, theta, omega = state
    new_state = np.zeros_like(state)

    # Evitiamo la divisione per zero se il pesce va dritto
    if abs(omega) < 0.0001:
        new_state[0] = x + v * dt * np.cos(theta)
        new_state[1] = y + v * dt * np.sin(theta)
    else:
        # Equazioni non lineari del moto circolare
        new_state[0] = x + (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta))
        new_state[1] = y + (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta))

    new_state[2] = v                   # Velocità costante
    new_state[3] = theta + omega * dt  # Nuovo angolo
    new_state[4] = omega               # Velocità angolare costante
    
    return new_state

def measurement_function(state):
    """
    La camera ad eventi ci restituisce solo la posizione (x, y).
    """
    return np.array([state[0], state[1]])
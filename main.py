#State
# Il numero di entry di ogni possibile stato è 256, dati dal prodotto 16x4x4. 16 è dovuto al fatto che ogni riquadro può
# contenere 0 o un multiplo di 2 (2^i con 0<i<16). In base allo stato l'entry avrà valore 0 o 1.

#Action
# Le azioni possibili sono quattro, ovvero: destra, sinistra, su e giù. Tuttavia non tutte le azioni sono disponibili in
# ogni stato.

#Reward
# Il premio è dato dalla somma dei valori di tutte le celle fuse dopo l'azione.
#from agent import QLearningAgent
from env import Game2048Env


def play_game():
    # Inizializza l'ambiente di gioco
    env = Game2048Env()

    print("Benvenuto a 2048!")
    print("Usa le seguenti tasti per muoverti: W (su), S (giù), A (sinistra), D (destra)")
    print("Per uscire dal gioco, premi Q.")

    while True:
        # Stampa lo stato corrente del gioco
        env.render()

        # Ottieni l'input dell'utente
        action = input("Scegli una mossa (W/A/S/D): ").upper()

        # Verifica l'input dell'utente
        if action == 'Q':
            print("Grazie per aver giocato. Arrivederci!")
            break
        elif action in ['W', 'A', 'S', 'D']:
            # Mappa l'input dell'utente a un'azione nell'intervallo [0, 1, 2, 3]
            action_mapping = {'W': 0, 'A': 2, 'S': 1, 'D': 3}
            action = action_mapping[action]

            # Esegui la mossa nell'ambiente di gioco
            state, reward, done, _ = env.step(action)

            # Verifica se il gioco è terminato
            if done:
                env.render()
                print("Il gioco è terminato. Punteggio finale:", reward)
                break
        else:
            print("Input non valido. Usa W/A/S/D per muoverti o Q per uscire.")


if __name__ == '__main__':
    play_game()



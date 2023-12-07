
#State
# Il numero di entry di ogni possibile stato è 256, dati dal prodotto 16x4x4. 16 è dovuto al fatto che ogni riquadro può
# contenere 0 o un multiplo di 2 (2^i con 0<i<16). In base allo stato l'entry avrà valore 0 o 1.

#Action
# Le azioni possibili sono quattro, ovvero: destra, sinistra, su e giù. Tuttavia non tutte le azioni sono disponibili in
# ogni stato.

#Reward
# Il premio è dato dalla somma dei valori di tutte le celle fuse dopo l'azione.
from agent import QLearningAgent
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

def play_agent():
    env = Game2048Env()
    agent = QLearningAgent(state_size=2048, action_size=4)

    # Esegui l'addestramento per un certo numero di episodi
    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.actuate(state)
            print(action)
            next_state, reward, done, _ = env.step(action)
            agent.percept(state, action, next_state, reward)

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
# Avvia il gioco

if __name__ == '__main__':
    play_game()



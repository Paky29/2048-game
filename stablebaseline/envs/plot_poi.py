import matplotlib
import matplotlib.pyplot as plt
from IPython import display
matplotlib.use('Agg')

class RealTimeScorePlotter:
    def __init__(self):
        self.scores = []
        self.iterations = []

        plt.ion()  # Abilita il modo interattivo
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Iterazioni')
        self.ax.set_ylabel('Punteggio')
        self.ax.set_title('Punteggio per Iterazione')

    def update_plot(self, new_score):
        self.scores.append(new_score)
        self.iterations.append(len(self.scores))

        self.ax.clear()
        self.ax.plot(self.iterations, self.scores, label='Punteggio')
        self.ax.legend()
        self.ax.set_xlabel('Iterazioni')
        self.ax.set_ylabel('Punteggio')
        self.ax.set_title('Punteggio per Iterazione')

        plt.draw()
        plt.pause(0.1)  # Breve pausa per permettere l'aggiornamento del grafico

        # Simula un processo di calcolo (sostituisci con il tuo codice di addestramento)
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def close_plot(self):
        plt.savefig('punteggio_addestramento.png')
        plt.ioff()
        display.clear_output(wait=True)

    def save(self):
        plt.savefig('punteggio_addestramento.png')
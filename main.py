# import modules
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





class Agent:

    def __init__(self, T):
        self.MW_bandit = False
        self.MW_expert = False
        self.ucb = False

        # load data from CSV file
        with open('Milano_timeseries.csv', 'r', encoding='ISO-8859-1') as csvfile:
            reader = csv.reader(csvfile)
            self.data = np.array(list(reader)).astype('float')

        # enviroment parameters
        self.T = T  # Horizon
        self.n_servers = self.data.shape[0]  # number of servers = rows of csv
        self.htta =  np.sqrt(np.log(self.n_servers) / self.T)  # htta for discount factor
        self.weights = np.ones(self.n_servers)  # array of 1's divides by number of servers to get same weights
        self.losses = np.zeros(self.n_servers)  # array to store the losses of each server for expert enviroment
        self.regret = []
        self.root = []

    def grapher(self, regret1):
        plt.plot(np.arange(1, self.T + 1), regret1)
        plt.title("Hedge in expert [T = %d]" % self.T)
        plt.xlabel("Round T")
        plt.ylabel("Regret")
        plt.show()

        # plt.plot(np.arange(1, self.T + 1), regret_ucb)
        # plt.title("UCB Performance [T = %d, k = %d]" % (self.T, self.k))
        # plt.xlabel("Round T")
        # plt.ylabel("Regret")
        # plt.show()
        #
        # plt.plot(np.arange(1, self.T + 1), self.regret, color='r', label='MW')
        # plt.plot(np.arange(1, self.T + 1), self.root, color='b', label='sqrt(T)')
        # plt.title("e-Greedy and UCB common plot [T = %d]" % (self.T))
        # plt.xlabel("Round T")
        # plt.ylabel("Regret")
        #
        # plt.legend()
        # plt.show()


    def mw_expert(self):
        for i in range(self.T):
            probabilities = self.weights / np.sum(self.weights)  # propabilities for prediction
            prediction = np.random.choice(self.n_servers, p=probabilities)  # randomised prediction based on weights

            # get the smaller delay
            best_server = best_server = np.argmin(self.data[:, i])
            best_delay = self.data[best_server, i]
            # calculate the loss of each expert
            for j in range(self.n_servers):
                delay = self.data[j, i]
                self.losses[j] = np.abs(delay - best_delay)
                if j == prediction:
                    self.regret.append(self.losses[j])

            # Update the weights based on the expert losses and the learning rate
            for k in range(self.n_servers):
                self.weights[k] *= np.power((1 - self.htta), self.losses[k])
            self.weights /= np.sum(self.weights)

            self.root.append(np.sqrt(i))
        return np.cumsum(self.regret)

    def run(self):
        if self.MW_expert:
            mw_regret = self.mw_expert()
            self.grapher(mw_regret)
        exit()

if __name__ == '__main__':
    agent = Agent(7000)
    agent.MW_expert = True
    agent.run()


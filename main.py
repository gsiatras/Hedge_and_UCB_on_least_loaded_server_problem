# import modules
import csv
import numpy as np
import matplotlib.pyplot as plt




class Agent:

    def __init__(self, T):
        self.bandit = False
        self.expert = False
        self.ucb = False

        # load data from CSV file
        with open('Milano_timeseries.csv', 'r', encoding='ISO-8859-1') as csvfile:
            reader = csv.reader(csvfile)
            self.data = np.array(list(reader)).astype('float')

        # enviroment parameters
        self.T = T  # Horizon
        self.n_servers = self.data.shape[0]  # number of servers = rows of csv
        self.eta = np.sqrt(np.log(self.n_servers) / self.T)  # eta for discount factor
        self.weights = np.ones(self.n_servers)  # array of 1's divides by number of servers to get same weights
        self.losses = np.zeros(self.n_servers)  # array to store the losses of each server for expert enviroment
        self.regret = []  # array to store regret
        self.probabilities = np.ones(self.n_servers)  # array to store propability of choosing each server
        self.server_ucb = np.full(self.n_servers, np.inf)  # array of upper confidence bound of each server set to inf so we explore atleast once
        self.server_pulls = np.zeros(self.n_servers)  # array of server pulls of each server
        self.server_losses = np.zeros(self.n_servers)
    def reset(self):
        self.weights = np.ones(self.n_servers)
        self.regret = []
        self.probabilities = np.ones(self.n_servers)
    def grapher(self, regret1, regret2, regret3):
        # plt.plot(np.arange(1, self.T + 1), regret1)
        # plt.title("Hedge in expert [T = %d]" % self.T)
        # plt.xlabel("Round T")
        # plt.ylabel("Regret")
        # plt.show()
        #
        # plt.plot(np.arange(1, self.T + 1), regret2)
        # plt.title("Hedge in bandit [T = %d]" % self.T)
        # plt.xlabel("Round T")
        # plt.ylabel("Regret")
        # plt.show()

        plt.plot(np.arange(1, self.T + 1), regret1, color='r', label='MW in expert')
        plt.plot(np.arange(1, self.T + 1), regret2, color='b', label='MW in bandit')
        plt.title("Multiplicative Weights Algorithm [T = %d]" % self.T)
        plt.xlabel("Round T")
        plt.ylabel("Regret")

        plt.legend()
        plt.show()

        plt.plot(np.arange(1, self.T + 1), regret2, color='r', label='MW in bandit')
        plt.plot(np.arange(1, self.T + 1), regret3, color='b', label='UCB')
        plt.title("Multiplicative-UCB in bandit [T = %d]" % self.T)
        plt.xlabel("Round T")
        plt.ylabel("Regret")

        plt.legend()
        plt.show()

    def decide(self):
        if self.expert:
            self.probabilities = self.weights / np.sum(self.weights)  # probabilities for prediction
        if self.bandit:
            self.probabilities = (1-self.eta)*(self.weights / np.sum(self.weights)) + self.eta/self.n_servers

        prediction = np.random.choice(self.n_servers, p=self.probabilities)  # randomised prediction based on weights
        return prediction
    def mw_run(self):
        for i in range(self.T):
            # self.eta = np.sqrt(np.log(self.n_servers) / (i+1))
            # get the server prediction
            prediction = self.decide()
            # get the smaller delay
            best_server = best_server = np.argmin(self.data[:, i])
            best_delay = self.data[best_server, i]

            if self.expert:
                # calculate the loss of each expert
                for j in range(self.n_servers):
                    delay = self.data[j, i]
                    self.losses[j] = np.abs(delay - best_delay)
                    if j == prediction:
                        self.regret.append(self.losses[j])  # regret for based on our prediction

                # Update the weights based on the expert losses
                for k in range(self.n_servers):
                    self.weights[k] *= np.power((1 - self.eta), self.losses[k])
                self.weights /= np.sum(self.weights)  # normalise the weights

            if self.bandit:
                # calculate loss of predicted server
                delay = self.data[prediction, i]
                loss = np.abs(delay - best_delay)
                self.regret.append(loss)

                # update the weights
                new_loss = loss/self.probabilities[prediction]
                self.weights[prediction] *= np.power((1 - self.eta), new_loss)
                self.weights /= np.sum(self.weights)  # normalise the weights

        return np.cumsum(self.regret)

    def ucb_run(self):
        for i in range(self.T):
            decision = np.argmax(self.server_ucb)
            # get the smaller delay
            best_server = np.argmin(self.data[:, i])
            best_delay = self.data[best_server, i]
            # get servers delay and loss
            delay = self.data[decision, i]
            loss = np.abs(delay - best_delay)
            self.server_pulls[decision] += 1
            # calculate total loss
            if self.server_ucb[decision] == 0:
                self.server_losses[decision] += 1-loss
            else:
                self.server_losses[decision] = ((self.server_losses[decision]*self.server_pulls[decision] - 1)
                                                + 1-loss) / (self.server_pulls[decision])

            self.server_ucb[decision] = self.server_losses[decision] + np.sqrt(2 * np.log(i+1) / self.server_pulls[decision])
            # add regret
            self.regret.append(loss)
        return np.cumsum(self.regret)


    def run(self):
        self.expert = True
        mw_ex_regret = self.mw_run()

        self.reset()
        self.expert = False
        self.bandit = True
        mw_bd_regret = self.mw_run()

        self.reset()
        self.ucb = True
        ucb_regret = self.ucb_run()
        self.grapher(mw_ex_regret, mw_bd_regret, ucb_regret)




if __name__ == '__main__':
    agent = Agent(7000)
    agent.run()
    exit()


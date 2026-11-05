import numpy as np


class Environment:
    def __init__(self):
        pass

class HouseSellingEnv(Environment):
    def __init__(self, horizon=10, cost=10, max_offer=100):
        self.horizon = horizon
        self.cost = cost
        self.max_offer = max_offer
        self.states = [i for i in range(max_offer + 1)]
        self._init_probabilities()
        self.actions = [0, 1]  # 0: Accept, 1: Reject

    def reset(self):
        return np.random.choice(self.states)

    def _init_probabilities(self):
        self.P = np.ones(self.max_offer + 1)
        mu = self.max_offer / 2
        sigma = np.sqrt(self.max_offer / 4)
        for i in range(self.max_offer + 1):
            self.P[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((i - mu) ** 2) / (2 * sigma**2)
            )
        self.P /= np.sum(self.P)
        # self.P=np.ones(self.max_offer+1)
        # self.P/=np.sum(self.P)

    def get_next_state(self):
        return np.random.choice(self.states, p=self.P)

    def generate_sequence(self):
        sequence = [self.reset()]
        for i in range(1, self.horizon):
            sequence.append(self.get_next_state())
        return sequence

    def exp_max(self, n, x):
        vals = np.array([max(x, i) for i in range(self.max_offer + 1)])
        return np.dot(vals, self.P)

    def exp(self, n):
        return self.max_offer / 2

    def cdf(self):
        return np.array([np.sum(self.P[: i + 1]) for i in range(self.max_offer + 1)])

    def exp_indictor_less_than(self, x):
        vals = np.array([i if i <= x else 0 for i in range(self.max_offer + 1)])
        return np.dot(vals, self.P)

    def f_vals(self):
        f_vals = np.zeros((self.horizon + 1))
        cdf_vals = self.cdf()
        mu = self.max_offer // 2
        for i in range(1, self.horizon + 1):
            f_vals[i] = (mu - self.cost + f_vals[i - 1]) * cdf_vals[
                min(int(mu - self.cost + f_vals[i - 1]), self.max_offer)
            ] - self.exp_indictor_less_than(mu - self.cost + f_vals[i - 1])
        return f_vals


class RandomWalk(Environment):
    def __init__(self, horizon=10, cost=10, max_offer=100, p=0.5, step=10):
        self.horizon = horizon
        self.cost = cost
        self.max_offer = max_offer
        self.p = p
        self.step = step
        self.states = [i for i in range(max_offer + 1)]
        self.actions = [0, 1]

    def reset(self):
        return np.random.choice(self.states)

    def get_next_state(self, current_state):
        if np.random.rand() < self.p:
            return min(current_state + self.step, self.max_offer)
        else:
            return max(current_state - self.step, 0)

    def generate_sequence(self):
        sequence = [self.reset()]
        for i in range(1, self.horizon):
            sequence.append(self.get_next_state(sequence[-1]))
        return sequence

class SingleOptimalEnvironment:
    def __init__(self, N=100, C=10, alpha=1):
        self.N = N  # number of states
        self.C = C  # holding cost
        self.S = [i for i in range(N + 1)]  # states
        self.A = ["hold", "sell"]  # actions
        self.P = self._init_probabilities()  # probabilities of offers
        self.Psum = self.sum_prob()
        self.XPsum = self.sum_exp()
        self.TPM = self._init_TPM()
        self.i_star = self.find_state()
        self.alpha = alpha
        self.V = self.update_values()

    def sum_prob(self):
        A = np.zeros(self.N + 1)
        A[0] = self.P[0]
        for i in range(1, self.N + 1):
            A[i] = A[i - 1] + self.P[i]
        return A

    def sum_exp(self):
        A = np.zeros(self.N + 1)
        for i in range(1, self.N + 1):
            A[i] = A[i - 1] + i * self.P[i]
        return A

    def find_state(self):
        for i in range(self.N + 1):
            if self.C >= self.XPsum[self.N] - self.XPsum[i] - i * (
                self.Psum[self.N] - self.Psum[i]
            ):
                return i
        return self.N

    def _init_probabilities(self):
        P = np.zeros(self.N + 1)
        mu = self.N / 2
        sigma = np.sqrt(self.N / 4)
        for i in range(self.N + 1):
            P[i] = self._gaussian_prob(i, mu, sigma)
        return P / np.sum(P)  # Normalization

    def _init_TPM(self):
        TPM = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                if j < i:
                    TPM[i][j] = 0
                elif i == j:
                    TPM[i][j] = self.Psum[i]
                else:
                    TPM[i][j] = self.P[j]
        return TPM

    @staticmethod
    def _gaussian_prob(i, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((i - mu) ** 2) / (2 * sigma**2)
        )

    def reward(self, s, a):
        if a == "sell":
            return s
        return -self.C

    def get_next_state(self, s, a):
        if a == "sell":
            return "terminal"
        return max(s, np.random.choice(self.S, p=self.P))

    def step(self, s, a):
        state = self.get_next_state(s, a)
        reward = self.reward(s, a)
        done = state == "terminal"
        return state, reward, done

    def update_values(self):
        V = np.zeros(self.N + 1)
        R = np.zeros(self.N + 1)
        P = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.i_star):
            R[i] = -self.C
        for i in range(self.i_star, self.N + 1):
            R[i] = i

        for i in range(self.N):
            for j in range(self.N):
                if i < self.i_star:
                    P[i][j] = self.TPM[i][j]
                else:
                    P[i][j] = 0

        I = np.identity(self.N + 1)
        V = np.dot(np.linalg.inv(I - self.alpha * P), R)
        return V

    def get_value(self, s, a):
        if a == "hold":
            return self.V[s]
        return s

    def transition_probability(self, i, j, a):
        if a == "hold":
            return self.TPM[i][j]
        return 1 if j == self.N + 1 else 0

import numpy as np


class MarkovianHouseSellingEnv:
    def __init__(self, horizon=10, cost=10, max_offer=100, alpha=1.0):
        self.horizon = horizon
        self.cost = cost
        self.max_offer = max_offer
        self.states = [i for i in range(max_offer + 1)]
        self.actions = [0, 1]  # 0: Accept, 1: Reject
        self.alpha = alpha
        self._init_probabilities()
        self._init_TPM()

    def reset(self):
        return np.random.choice(self.states, p=self.P)

    def _init_probabilities(self):
        self.P = np.zeros(self.max_offer + 1)
        mu = self.max_offer / 2
        sigma = np.sqrt(self.max_offer / 4)
        for i in range(self.max_offer + 1):
            self.P[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -((i - mu) ** 2) / (2 * sigma**2)
            )
        self.P /= np.sum(self.P)

    def _init_TPM(self):
        """Transition matrix TPM[s][s'] = P(s' | s)"""
        self.TPM = np.zeros((self.max_offer + 1, self.max_offer + 1))
        self.P_cumsum = np.cumsum(self.P)

        for s in range(self.max_offer + 1):
            for s_prime in range(self.max_offer + 1):
                if s_prime < s:
                    self.TPM[s, s_prime] = 0
                elif s_prime == s:
                    self.TPM[s, s_prime] = self.P_cumsum[s]
                else:
                    self.TPM[s, s_prime] = self.P[s_prime]

            self.TPM[s] /= np.sum(self.TPM[s])  # Normalize each row

    def get_next_state(self, s):
        return np.random.choice(self.states, p=self.TPM[s])

    def generate_sequence(self):
        sequence = [self.reset()]
        for _ in range(1, self.horizon):
            next_state = self.get_next_state(sequence[-1])
            sequence.append(next_state)
        return sequence

    def exp_max(self, n, x, s):
        """
        Conditional expected maximum:
        E_s[max(x, S_{t+1}) | S_t = s] = sum_{s'} max(x, s') * P(s' | s)
        """
        vals = np.array([max(x, s_prime) for s_prime in self.states])
        return np.dot(vals, self.TPM[s])

    # Optional utilities if needed
    def cdf(self):
        return np.cumsum(self.P)

    def exp_indictor_less_than(self, x):
        vals = np.array([i if i <= x else 0 for i in self.states])
        return np.dot(vals, self.P)

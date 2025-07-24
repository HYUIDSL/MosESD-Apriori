import numpy as np
import scipy.stats as stats


# Trend Residual Vector


class TRES:  # Trend residual vector
    def __init__(self, data, time=None, wins=None):
        self.data = data[:wins]
        self.time = list(range(1, wins + 1)) if time is None else time[:wins]
        self.original_data = data
        self.original_time = time
        self.tres = []
        self.x_bar = np.mean(self.time)
        self.y_bar = np.mean(self.data)
        self.wins = wins
        self._initialize()

    def _initialize(self):
        beta = sum(
            [
                (self.time[i] - self.x_bar) * (self.data[i] - self.y_bar)
                for i in range(self.wins)
            ]
        ) / sum([(t - self.x_bar) ** 2 for t in self.time])
        alpha = self.y_bar - beta * self.x_bar
        self.tres.append(
            self.data[self.wins - 1] - (alpha + beta * self.time[self.wins - 1])
        )
        for i in range(self.wins, len(self.original_data)):
            self.data.pop(0)
            self.time.pop(0)
            self.data.append(self.original_data[i])
            self.time.append(self.original_time[i])
            self.x_bar -= (
                self.original_time[i - self.wins] - self.original_time[i]
            ) / self.wins
            self.y_bar -= (
                self.original_data[i - self.wins] - self.original_data[i]
            ) / self.wins
            beta = sum(
                [
                    (self.time[j] - self.x_bar) * (self.data[j] - self.y_bar)
                    for j in range(self.wins)
                ]
            ) / sum([(t - self.x_bar) ** 2 for t in self.time])
            alpha = self.y_bar - beta * self.x_bar
            self.tres.append(self.data[-1] - (alpha + beta * self.time[-1]))

    def update(self, ond, ont=None):  # ond: observed new data, ont: observed new time
        first_data = self.data.pop(0)
        first_time = self.time.pop(0)
        self.data.append(ond)
        self.time.append(self.time[-1] + 1 if ont is None else ont)
        self.x_bar -= (first_time - self.time[-1]) / self.wins
        self.y_bar -= (first_data - ond) / self.wins
        beta = sum(
            [
                (self.time[i] - self.x_bar) * (self.data[i] - self.y_bar)
                for i in range(self.wins)
            ]
        ) / sum([(t - self.x_bar) ** 2 for t in self.time])
        alpha = self.y_bar - beta * self.x_bar
        tres_ = ond - (alpha + beta * self.time[-1])
        self.tres.append(tres_)

        return tres_

    def replace(self, rep):
        prev = self.data[self.wins - 1]
        self.data[self.wins - 1] = rep
        self.y_bar -= (prev - rep) / self.wins


# Trend change rate vector


class TCHA:
    def __init__(self, data, wins, time=None):
        if time is None:
            time = list(range(1, len(data) + 1))
        self.data = data[(len(data) - wins) :]
        self.time = time[(len(time) - wins) :]
        tcha_data = []
        for x, y in zip(data[wins - 1 :], data[: len(data) - wins + 1]):
            tcha_data.append(x - y)
        tcha_time = []
        for x, y in zip(time[wins - 1 :], time[: len(time) - wins + 1]):
            tcha_time.append(x - y)
        tcha = []
        for x, y in zip(tcha_data, tcha_time):
            tcha.append(x / y)
        self.tcha = tcha
        self.wins = wins

    def update(self, ond, ont=None):
        if ont is None:
            ont = self.time[self.wins] + 1
        self.data = self.data[1:] + [ond]
        self.time = self.time[1:] + [ont]
        tcha_ = (self.data[self.wins - 1] - self.data[0]) / (
            self.time[self.wins - 1] - self.time[0]
        )
        self.tcha = self.tcha[1:] + [tcha_]
        return tcha_

    def replace(self, rep):
        self.data[self.wins - 1] = rep


# Sequential Extreme studentized deviate test for trend residual vector
class SESD_tres:
    def __init__(self, data=None, alpha=0.01, maxr=10):
        self.mean = 0
        self.sqsum = 0
        self.alpha = alpha
        self.maxr = maxr
        self.data = data
        self.size = len(data)
        self.mean = np.mean(data)
        self.sqsum = np.sum(np.square(data))
        lambdas = [0, 0, 0]
        for i in range(3, len(data) + 1):
            lambdas.append(self.get_lambda(alpha, i))
        self.lambdas = lambdas

    def test(self, on):
        out = self.data[0]
        self.data = np.append(self.data[1:], on)
        self.mean += -(out - on) / self.size
        self.sqsum += (-(out**2)) + (on**2)
        mean_ = self.mean
        sqsum_ = self.sqsum
        size_ = self.size
        data_ = self.data
        sd_ = np.sqrt((sqsum_ - size_ * (mean_**2) + 1e-8) / (size_ - 1))
        ares = np.abs((data_ - mean_) / sd_)
        esd_index = np.argmax(ares)
        esd = ares[esd_index]
        try:
            if esd > self.lambdas[size_]:
                if esd_index == size_ - 1:
                    return esd
            else:
                return 0
        except:
            return 0

        for i in range(2, self.maxr + 1):
            size_ -= 1
            mean_ = ((size_ + 1) * mean_ - data_[esd_index]) / size_
            sqsum_ -= data_[esd_index] ** 2
            sd_ = np.sqrt((sqsum_ - size_ * mean_**2 + 1e-8) / (size_ - 1))

            data_ = np.delete(data_, esd_index)
            ares = np.abs((data_ - mean_) / sd_)
            esd_index = np.argmax(ares)
            esd = ares[esd_index]
            try:
                if esd > self.lambdas[size_]:
                    if esd_index == size_ - 1:
                        return esd
                else:
                    return 0
            except:
                return 0
        return 0

    def get_lambda(self, alpha, size):
        t = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        lmbda = t * (size - 1) / np.sqrt((size + t**2) * size)
        return lmbda


# Sequential Extreme studentized deviate test for trend change rate vector
class SESD_tcha:
    def __init__(self, data=None, alpha=0.01, maxr=10):
        self.mean = 0
        self.sqsum = 0
        self.alpha = alpha
        self.maxr = maxr
        self.data = data
        self.size = len(data)
        self.mean = np.mean(data)
        self.sqsum = np.sum(np.square(data))
        lambdas = [0, 0, 0]
        for i in range(3, len(data) + 1):
            lambdas.append(self.get_lambda(alpha, i))
        self.lambdas = lambdas

    def test(self, on):
        out = self.data[0]
        self.data = np.append(self.data[1:], on)
        self.mean += -(out - on) / self.size
        self.sqsum += (-(out**2)) + (on**2)
        mean_ = self.mean
        sqsum_ = self.sqsum
        size_ = self.size
        data_ = self.data
        sd_ = np.sqrt((sqsum_ - size_ * (mean_**2) + 1e-8) / (size_ - 1))
        ares = np.abs((data_ - mean_) / sd_)
        esd_index = np.argmax(ares)
        esd = ares[esd_index]
        try:
            if esd > self.lambdas[size_]:
                if esd_index == size_ - 1:
                    return esd
            else:
                return 0
        except:
            return 0

        for i in range(2, self.maxr + 1):
            size_ -= 1
            mean_ = ((size_ + 1) * mean_ - data_[esd_index]) / size_
            sqsum_ -= data_[esd_index] ** 2
            sd_ = np.sqrt((sqsum_ - size_ * mean_**2 + 1e-8) / (size_ - 1))
            data_ = np.delete(data_, esd_index)
            ares = np.abs((data_ - mean_) / sd_)
            esd_index = np.argmax(ares)
            esd = ares[esd_index]
            try:
                if esd > self.lambdas[size_]:
                    if esd_index == size_ - 1:
                        return esd
                else:
                    return 0
            except:
                return 0
        return 0

    def get_lambda(self, alpha, size):
        t = stats.t.ppf(1 - alpha / (2 * size), size - 2)
        lmbda = t * (size - 1) / np.sqrt((size + t**2) * size)
        return lmbda

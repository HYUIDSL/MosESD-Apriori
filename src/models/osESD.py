from .osESD_components import TCHA, TRES, SESD_tcha, SESD_tres


class osESD:
    def __init__(
        self, data, dwins, rwins, init_size, alpha, maxr, condition, time=None
    ):

        if time is None:
            time = list(range(1, len(data) + 1))

        self.init_data = data[:init_size]
        self.online_data = data[init_size:]
        self.init_time = time[:init_size]
        self.online_time = time[init_size:]
        self.dwins = dwins
        self.rwins = rwins
        self.init_size = init_size
        self.alpha = alpha
        self.maxr = maxr
        self.condition = condition
        self.initiate()

    def initiate(self):
        c_ins = TCHA(data=self.init_data, time=self.init_time, wins=self.dwins)
        r_ins = TRES(data=self.init_data, time=self.init_time, wins=self.rwins)
        self.SESD_TCHA = SESD_tcha(
            data=c_ins.tcha.copy(), alpha=self.alpha, maxr=self.maxr
        )
        self.SESD_TRES = SESD_tres(
            data=r_ins.tres.copy(), alpha=self.alpha, maxr=self.maxr
        )
        self.c_ins = c_ins
        self.r_ins = r_ins

    def test_values(self, idx):
        c_val = self.SESD_TCHA.test(
            self.c_ins.update(self.online_data[idx], self.online_time[idx])
        )
        r_val = self.SESD_TRES.test(
            self.r_ins.update(self.online_data[idx], self.online_time[idx])
        )
        c_anom = 0 if c_val == 0 else 1
        r_anom = 0 if r_val == 0 else 1
        return c_val, r_val, c_anom, r_anom

    def check_values(self, c_anom, r_anom):
        if self.condition:
            function_ = c_anom and r_anom
        else:
            function_ = c_anom or r_anom
        if function_:
            D = self.r_ins.data.copy()
            T = self.r_ins.time.copy()
            del D[self.rwins - 1]
            del T[self.rwins - 1]
            x_bar = (
                (self.rwins * self.r_ins.x_bar) - self.r_ins.time[self.rwins - 1]
            ) / (self.rwins - 1)
            y_bar = (
                (self.rwins * self.r_ins.y_bar) - self.r_ins.data[self.rwins - 1]
            ) / (self.rwins - 1)
            beta_ = sum((T - x_bar) * (D - y_bar)) / sum((T - x_bar) ** 2)
            alpha_ = y_bar - beta_ * x_bar
            rep = alpha_ + beta_ * T[self.rwins - 2]
            self.c_ins.replace(rep)
            self.r_ins.replace(rep)
            return 1
        return 0

    def predict_idx(self, idx):  ### index is based on online_data! not total data
        canom = self.SESD_TCHA.test(
            self.c_ins.update(self.online_data[idx], self.online_time[idx])
        )
        ranom = self.SESD_TRES.test(
            self.r_ins.update(self.online_data[idx], self.online_time[idx])
        )
        if self.condition:
            function_ = canom and ranom
        else:
            function_ = canom or ranom
        if function_:
            D = self.r_ins.data.copy()
            T = self.r_ins.time.copy()
            del D[self.rwins - 1]
            del T[self.rwins - 1]
            x_bar = (
                (self.rwins * self.r_ins.x_bar) - self.r_ins.time[self.rwins - 1]
            ) / (self.rwins - 1)
            y_bar = (
                (self.rwins * self.r_ins.y_bar) - self.r_ins.data[self.rwins - 1]
            ) / (self.rwins - 1)
            beta_ = sum((T - x_bar) * (D - y_bar)) / sum((T - x_bar) ** 2)
            alpha_ = y_bar - beta_ * x_bar
            rep = alpha_ + beta_ * T[self.rwins - 2]
            self.c_ins.replace(rep)
            self.r_ins.replace(rep)
            return 1
        return 0

    def predict_all(self):
        anomaly_index = []
        for i in range(len(self.online_data)):
            canom = self.SESD_TCHA.test(
                self.c_ins.update(self.online_data[i], self.online_time[i])
            )
            ranom = self.SESD_TRES.test(
                self.r_ins.update(self.online_data[i], self.online_time[i])
            )
            if self.condition:
                function_ = canom and ranom
            else:
                function_ = canom or ranom
            if function_:
                anomaly_index.append(i + self.init_size)
                D = self.r_ins.data.copy()
                T = self.r_ins.time.copy()
                del D[self.rwins - 1]
                del T[self.rwins - 1]
                x_bar = (
                    (self.rwins * self.r_ins.x_bar) - self.r_ins.time[self.rwins - 1]
                ) / (self.rwins - 1)
                y_bar = (
                    (self.rwins * self.r_ins.y_bar) - self.r_ins.data[self.rwins - 1]
                ) / (self.rwins - 1)
                beta_ = sum((T - x_bar) * (D - y_bar)) / sum((T - x_bar) ** 2)
                alpha_ = y_bar - beta_ * x_bar
                rep = alpha_ + beta_ * T[self.rwins - 2]
                self.c_ins.replace(rep)
                self.r_ins.replace(rep)

        return anomaly_index

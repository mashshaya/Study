import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.s0 = 1.0
        self.p = 0.5

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * np.power((self.s0 / (self.s0 + iteration)), self.p)


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        delta_w = self.update_weights()
        self.iteration += 1
        return delta_w


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):

        X_train = self.model.X_train
        y_train = self.model.y_train

        gradient = self.model.compute_gradients(X_train, y_train)
        η = self.lr_schedule.get_lr(self.iteration)  

        delta_w = -η * gradient
        self.model.w = self.model.w + delta_w
        return delta_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):

        X_train = self.model.X_train
        y_train = self.model.y_train
        n_samples = X_train.shape[0]

        batch_idx = np.random.randint(0, n_samples, size=self.batch_size)
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        gradient = self.model.compute_gradients(X_batch, y_batch)
        η = self.lr_schedule.get_lr(self.iteration)

        delta_w = -η * gradient
        self.model.w = self.model.w + delta_w
        return delta_w


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_avg = None

    def update_weights(self):

        X_train = self.model.X_train
        y_train = self.model.y_train
        n_objects, n_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((n_objects, n_features))
            self.grad_avg = np.zeros(n_features)

        j = np.random.randint(0, n_objects)
        gradient_new = self.model.compute_gradients(X_train[j:j+1], y_train[j:j+1]).flatten()
        gradient_old = self.grad_memory[j].copy()

        self.grad_memory[j] = gradient_new
        self.grad_avg = self.grad_avg + (gradient_new - gradient_old) / n_objects 
        η = self.lr_schedule.get_lr(self.iteration)

        delta_w = -η * self.grad_avg
        self.model.w = self.model.w + delta_w
        return delta_w


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, alpha=0.9):
        super().__init__(lr_schedule)
        self.alpha = alpha
        self.velocity = None

    def update_weights(self):

        X_train = self.model.X_train
        y_train = self.model.y_train

        gradient = self.model.compute_gradients(X_train, y_train)
        η = self.lr_schedule.get_lr(self.iteration)

        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)

        self.velocity = self.alpha * self.velocity + η * gradient 

        delta_w = -self.velocity
        self.model.w = self.model.w + delta_w
        return delta_w


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):

        X_train = self.model.X_train
        y_train = self.model.y_train

        gradient = self.model.compute_gradients(X_train, y_train)
        η = self.lr_schedule.get_lr(self.iteration)

        if self.m is None:
            self.m = np.zeros_like(gradient)

        if self.v is None:
            self.v = np.zeros_like(gradient)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient) ** 2

        m_est = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_est = self.v / (1 - self.beta2 ** (self.iteration + 1))

        delta_w = -(η * m_est) / (np.sqrt(v_est) + self.eps)
        self.model.w = self.model.w + delta_w
        return delta_w        




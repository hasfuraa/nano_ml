from abc import ABC, abstractmethod
import numpy as np

class Op(ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    @abstractmethod
    def backward(self, dz: np.ndarray) -> np.ndarray:
        raise NotImplemented()

class Linear(Op):
    def __init__(self, in_features: int, out_features: int) -> None:
        self.W = np.random.rand(in_features, out_features)
        self.b = np.random.rand(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        z = x @ self.W + self.b
        return z

    def backward(self, dz: np.ndarray) -> np.ndarray:
        # W -- dz: (B, out); x: (B, in) -> W: (in, out)
        # b -- dz: (B, out); -> b: (out)
        # dx -- dz: (B, out); W: (in, out) -> dx: (B, in)
        dx = dz @ self.W.T
        dW = self.x.T @ dz
        db = dz.sum(axis=0)
        self.W -= dW
        self.b -= db
        return dx


class ReLU(Op):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, dz: np.ndarray) -> np.ndarray:
        mask = self.x >= 0
        return dz * mask

class Loss(ABC):
    def __call__(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(a, y)

    @abstractmethod
    def forward(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplemented()

    @abstractmethod
    def backward(self, a: np.ndarray, y: np.ndarray, lr: float) -> np.ndarray:
        raise NotImplemented()


class L2Loss(Loss):
    def forward(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert a.ndim == 2 and y.ndim == 2
        loss = np.linalg.norm(y - a, axis=1).mean(axis=0)
        return loss

    def backward(self, a: np.ndarray, y: np.ndarray, lr: float) -> np.ndarray:
        return (2 * lr * (a - y)) / a.shape[0]

class Sequential(object):
    def __init__(self, ops: list[Op]) -> None:
        self.ops = ops

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for op in self.ops:
            x = op(x)
        return x

    def backward(self, dz: np.ndarray) -> None:
        for op in self.ops[::-1]:
            dz = op.backward(dz)

class MLP(Sequential):
    def __init__(self, n_layers: int, in_size: int, out_size: int, hidden_size: int) -> None:
        ops = []
        for layer_idx in range(n_layers):
            i_size = in_size if layer_idx == 0 else hidden_size
            o_size = out_size if layer_idx == n_layers - 1 else hidden_size
            linear = Linear(i_size, o_size)
            layer_ops = [linear]
            if layer_idx != n_layers - 1:
                layer_ops += [ReLU()]
            ops += layer_ops
        super().__init__(ops)

def main() -> None:
    in_size = 8
    lr = 1e-3
    mlp = MLP(n_layers=3, in_size=in_size, out_size=1, hidden_size=16)
    l2 = L2Loss()

    # data
    batch_size = 128
    xs = np.random.rand(batch_size, in_size)
    ys = xs.sum(axis=1, keepdims=True)
    
    # train loop
    n_epochs, epoch_per_log = int(1e3), int(1e2)
    for epoch_idx in range(n_epochs):
        out = mlp(xs)
        loss = l2(out, ys)
        loss_backward = l2.backward(out, ys, lr)
        mlp.backward(loss_backward)
        if epoch_idx % epoch_per_log == 0:
            print(f"{epoch_idx=}, {loss=}")

if __name__ == "__main__":
    main()

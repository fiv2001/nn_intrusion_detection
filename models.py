import abc
import torch
import joblib
import lightgbm as lgb
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from config import CONFIG


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class LGBMModel(BaseModel):
    def __init__(self, num_leaves, n_estimators):
        self.model = lgb.LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class PytorchModel(BaseModel):
    @abc.abstractmethod
    def build_model(self, **model_parameters):
        pass

    def __init__(self, loss_function, optimizer, lr, n_epochs, batch_size, **model_parameters):
        self.build_model(**model_parameters)

        self.loss_function = self.get_loss_function(loss_function)
        self.optimizer_class = self.get_optimizer_class(optimizer)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model.to(self.device)


    def get_loss_function(self, loss_function):
        match loss_function:
            case "CrossEntropy":
                return nn.CrossEntropyLoss()
            case "MSE":
                return nn.MSELoss()
            case _:
                raise ValueError("Unsupported loss function")

    def get_optimizer_class(self, optimizer):
        match optimizer:
            case "Adam":
                return optim.Adam
            case "SGD":
                return optim.SGD
            case _:
                raise ValueError("Unsupported optimizer")

    def get_batch(self, data, batch_idx):
        left_bound = batch_idx * self.batch_size
        right_bound = min((batch_idx + 1) * self.batch_size, data.shape[0])
        batch = data[left_bound:right_bound]
        return torch.from_numpy(batch)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def fit(self, X_train, y_train):
        print("Fit started")
        self.model.train()
        writer = SummaryWriter()
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        for epoch in range(self.n_epochs):
            accuracies = []
            losses = []
            print(f"Epoch: {epoch}")
            for batch_idx in range((X_train.shape[0] + self.batch_size - 1) // self.batch_size):
                X_batch, y_batch = self.get_batch(X_train, batch_idx).float(), self.get_batch(y_train, batch_idx).long()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_iter = batch_idx * self.batch_size + (X_train.shape[0]) * epoch
                writer.add_scalar('Loss', loss.item(), n_iter)
                losses.append(loss.item())
                pred_classes = torch.argmax(y_pred.detach().cpu(), dim=1)
                accuracy = (pred_classes == y_batch.detach().cpu()).float().mean()
                accuracies.append(accuracy)
                writer.add_scalar('Accuracy', accuracy, n_iter)
                writer.add_scalar('Epoch', epoch, n_iter)
            writer.add_scalar('Avg Loss', np.array(losses).mean(), X_train.shape[0] * epoch)
            writer.add_scalar('Avg Accuracy', np.array(accuracies).mean(), X_train.shape[0] * epoch)
        print("Fit finished")

    def predict(self, X_test):
        self.model.eval()
        result = []
        if not hasattr(self, "device"):
            self.device = False
        for batch_idx in range((X_test.shape[0] + self.batch_size - 1)// self.batch_size):
            X_batch = self.get_batch(X_test, batch_idx).float()
            X_batch = X_batch.to(self.device)
            y_pred = self.model(X_batch).detach().cpu()
            pred_classes = torch.argmax(y_pred, dim=1)
            result.append(pred_classes.numpy())
        return np.concatenate(result, axis=None)


class Perceptron(PytorchModel):
    def build_model(self, **model_parameters):
        layers = []
        input_size = model_parameters['input_size']
        for hidden in model_parameters['hidden_layers']:
            layers.append(nn.Linear(input_size, hidden))
            layers.append(nn.ReLU())
            input_size = hidden
        layers.append(nn.Linear(input_size, model_parameters['output_size']))
        layers.append(nn.Softmax(dim=0))
        self.model = nn.Sequential(*layers)


def create_model():
    match CONFIG.MODEL.model_name:
        case "lgbm":
            return LGBMModel(
                num_leaves=CONFIG.MODEL.LGBM.num_leaves,
                n_estimators=CONFIG.MODEL.LGBM.num_trees,
            )
        case "perceptron":
            return Perceptron(
                loss_function=CONFIG.MODEL.PERCEPTRON.loss_function,
                optimizer=CONFIG.MODEL.PERCEPTRON.optimizer,
                lr=CONFIG.MODEL.PERCEPTRON.lr,
                n_epochs=CONFIG.MODEL.PERCEPTRON.n_epochs,
                batch_size=CONFIG.MODEL.PERCEPTRON.batch_size,
                input_size=CONFIG.MODEL.PERCEPTRON.input_size,
                output_size=CONFIG.MODEL.PERCEPTRON.output_size,
                hidden_layers=CONFIG.MODEL.PERCEPTRON.hidden_layers,
            )
        case _:
            raise ValueError("Unsupported model name")

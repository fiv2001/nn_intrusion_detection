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

    def prepare_data(self, data):
        return data

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

    def fit(self, X_train, y_train, predict=False):
        X_train = self.prepare_data(X_train)
        self.model.train()
        writer = SummaryWriter()
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        result = []
        for epoch in range(self.n_epochs):
            accuracies = []
            losses = []
            print(f"Epoch: {epoch}")
            for batch_idx in range((X_train.shape[0] + self.batch_size - 1) // self.batch_size):
                X_batch, y_batch = self.get_batch(X_train, batch_idx).float(), self.get_batch(y_train, batch_idx).long()
#                if batch_idx == 1500:
#                    print(X_batch)
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.model(X_batch)

                n_iter = batch_idx * self.batch_size + (X_train.shape[0]) * epoch
                if not predict:
                    loss = self.loss_function(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    writer.add_scalar('Loss', loss.item(), n_iter)
                    losses.append(loss.item())

                pred_classes = torch.argmax(nn.LogSoftmax(dim=1)(y_pred.detach().cpu()), dim=1)
#                if (batch_idx == 1500):
#                    print(pred_classes)
                result.append(pred_classes.numpy())
                accuracy = (pred_classes == y_batch.detach().cpu()).float().mean()
                accuracies.append(accuracy)
                writer.add_scalar('Accuracy', accuracy, n_iter)
                writer.add_scalar('Epoch', epoch, n_iter)
            if not predict:
                writer.add_scalar('Avg Loss', np.array(losses).mean(), X_train.shape[0] * epoch)
            writer.add_scalar('Avg Accuracy', np.array(accuracies).mean(), X_train.shape[0] * epoch)
        print("Fit finished")
        return np.concatenate(result, axis=None)

    def predict(self, X_test):
        with torch.no_grad():
            X_test = self.prepare_data(X_test)
            self.model.eval()
            result = []
            for batch_idx in range((X_test.shape[0] + self.batch_size - 1) // self.batch_size):
                X_batch = self.get_batch(X_test, batch_idx).float()
#                if batch_idx == 1500:
#                    print(X_batch)
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                pred_classes = torch.argmax(nn.LogSoftmax(dim=1)(y_pred.detach().cpu()), dim=1)
#                if (batch_idx == 1500):
#                    print(pred_classes)
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
        self.model = nn.Sequential(*layers)


class RecurrentModule(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, lookback, recurrent_class):
        super().__init__()
        self.recurrent = recurrent_class(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear((lookback + 1) * hidden_size, output_size)
    def forward(self, x):
        x, _ = self.recurrent(x)
        x = self.linear(x.flatten(start_dim=1))
        #x = nn.Softmax(dim=0)(x)
        return x


class RecurrentModel(PytorchModel):
    def choose_recurrent_class(self, model_name):
        match CONFIG.MODEL.model_name:
            case "rnn":
                return nn.RNN
            case "lstm":
                return nn.LSTM
            case "gru":
                return nn.GRU
            case _:
                raise ValueError("Unsupported model name")

    def build_model(self, **model_parameters):
        recurrent_class = self.choose_recurrent_class(model_parameters['model_name'])
        self.model = RecurrentModule(
            input_size=model_parameters['input_size'],
            output_size=model_parameters['output_size'],
            hidden_size=model_parameters['hidden_size'],
            num_layers=model_parameters['num_layers'],
            lookback=model_parameters['lookback'],
            recurrent_class=recurrent_class,
        )
        self.input_size = model_parameters['input_size']

    def prepare_data(self, data):
        rows = []
        for i in range(data.shape[0]):
            row = np.stack(np.split(data[i], data.shape[1] // self.input_size))
            rows.append(row)
        return np.array(rows)


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
        case "lstm" | "gru" | "rnn":
            return RecurrentModel(
                loss_function=CONFIG.MODEL.RECURRENT.loss_function,
                optimizer=CONFIG.MODEL.RECURRENT.optimizer,
                lr=CONFIG.MODEL.RECURRENT.lr,
                n_epochs=CONFIG.MODEL.RECURRENT.n_epochs,
                batch_size=CONFIG.MODEL.RECURRENT.batch_size,
                input_size=CONFIG.MODEL.RECURRENT.input_size,
                output_size=CONFIG.MODEL.RECURRENT.output_size,
                hidden_size=CONFIG.MODEL.RECURRENT.hidden_size,
                num_layers=CONFIG.MODEL.RECURRENT.num_layers,
                lookback=CONFIG.GENERAL.lookback,
                model_name=CONFIG.MODEL.model_name,
            )
        case _:
            raise ValueError("Unsupported model name")

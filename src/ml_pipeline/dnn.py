import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from tqdm import tqdm

# для адаптации под api sklearn
from sklearn.base import BaseEstimator, ClassifierMixin


# адаптер для DNN с методами fit и transform
class DNNAdapter(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        # дефолтные параметры нужны для адаптации под sklearn
        random_state: int = 42,
        in_features: int = 12,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        test_size: float = 0.2,
        epochs: int = 5,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataset = None
        self.generator = torch.Generator().manual_seed(random_state)

        hidden_size_1 = 128
        hidden_size_2 = 64
        dropout_rate = 0.25

        self.model = nn.Sequential(
            nn.Linear(in_features, out_features=hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_2, 2),
        )

        self.model.to(self.device)

        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.calc_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

    def fit(self, X, y):
        dataset = MyDataset(X, y)

        # датасет разбивается на трейн и тест еще раз
        train_data, val_data = random_split(
            dataset, [1 - self.test_size, self.test_size], generator=self.generator
        )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        lr_list = []
        best_loss = None
        best_loss_threshold_to_save = 0.001
        epochs_without_loss_improve = 0
        epochs_patience = 20

        for epoch in range(self.epochs):
            # отображает прогресс бар
            train_loop = tqdm(train_loader, leave=False)

            running_train_loss = []
            true_answers = 0
            mean_train_loss = 0

            self.model.train()
            for batch_x, batch_y in train_loop:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # forward
                pred = self.model(batch_x)
                loss = self.calc_loss(pred, batch_y)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss.append(loss.item())
                mean_train_loss = sum(running_train_loss) / len(running_train_loss)

                # print(pred)
                # print(batch_y)
                # true_answers += (pred == batch_y).sum().item()

                train_loop.set_description(
                    f"{epoch}/{self.epochs}: mean train loss={mean_train_loss:.4f}"
                )

            # расчет значения метрики accuracy
            running_train_acc = true_answers / len(train_data)
            # сохранение значения лосса и метрик
            train_loss.append(mean_train_loss)
            train_acc.append(running_train_acc)

            mean_val_loss = 0

            # перевести модель в режим валидации
            self.model.eval()
            with torch.no_grad():
                running_val_loss = []
                true_answers = 0

                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.long().to(self.device)

                    # forward
                    pred = self.model(batch_x)
                    loss = self.calc_loss(pred, batch_y)

                    running_val_loss.append(loss.item())
                    mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                    pred_classes = torch.argmax(
                        torch.softmax(pred, dim=1), dim=1
                    ).float()

                    true_answers += (pred_classes == batch_y.float()).sum().item()

                    # обратного прохода и оптимизации нет

                running_val_acc = true_answers / len(val_data)
                # сохранение значения лосса и метрик
                val_loss.append(mean_val_loss)
                val_acc.append(running_val_acc)

            self.lr_scheduler.step(mean_val_loss)
            lr = self.lr_scheduler.get_last_lr()
            lr_list.append(lr[0])

            if best_loss is None:
                best_loss = mean_val_loss

            if mean_val_loss < best_loss - best_loss * best_loss_threshold_to_save:
                # обнулить кол-во эпох подряд без улучшения
                epochs_without_loss_improve = 0
                # обновить лосс
                best_loss = mean_val_loss

                # torch.save(
                #     self.model.state_dict(), f"model_state_dict_epoch_{epoch + 1}.pt"
                # )
                print(f"epoch {epoch}: saved model with loss {best_loss}")
            else:
                epochs_without_loss_improve += 1

            # в конце каждой эпохи выдавать промежуточные результаты
            print(
                f"Epoch #{epoch}/{self.epochs} ended: {mean_train_loss=:.4f}, {running_train_acc=:.4f}, {mean_val_loss=:.4f}, {running_val_acc=:.4f}"
            )

            # досрочно прервать обучение
            if epochs_without_loss_improve == epochs_patience:
                print(f"breaking on epoch {epoch}")
                break

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()


# можно еще использовать TensorDataset
class MyDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

        self.len_data = self.X.shape[0]

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.X[index], self.y[index]

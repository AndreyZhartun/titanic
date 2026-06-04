"""
Адаптер для DNN — регрессия (House Prices).
Использует MSELoss.

Все параметры конструктора идентичны DNNAdapter в классификации.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin


class DNNRegressorAdapter(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        *,
        # по описанию параметров см. также адаптер DNN титаника
        in_features: int = 301,
        hidden_sizes: list[int] | None = None,
        out_features: int = 1,
        dropout_rate: float = 0.25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 5,
        epochs_patience: int = 10,
        best_loss_threshold_to_save: float = 0.001,
        random_state: int = 42,
        # в титанике есть также директория для сохранения чекпоинтов save_dir
        # но в этом адаптере мы сохраняем чекпоинты просто в переменную, так проще
    ) -> None:
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 64]
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epochs_patience = epochs_patience
        self.best_loss_threshold_to_save = best_loss_threshold_to_save
        self.random_state = random_state

        self.val_dataset = None
        self.model_ = None
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_model(self) -> nn.Sequential:
        """
        Построить модель
        """

        sizes = [self.in_features, *self.hidden_sizes, self.out_features]
        layers = []

        for i in range(1, len(sizes)):
            # true - последний слой
            is_last = i == len(sizes) - 1
            # размеры предыдущего и текущего слоев
            in_size, out_size = sizes[i - 1], sizes[i]

            if is_last:
                layers.append(nn.Linear(in_size, out_size))
            else:
                layers.extend(
                    [
                        nn.Linear(in_size, out_size),
                        nn.BatchNorm1d(out_size),
                        nn.ReLU(),
                        nn.Dropout(self.dropout_rate),
                    ]
                )

        return nn.Sequential(*layers).to(self.device_)

    def fit(self, X, y):
        if self.val_dataset is None:
            raise ValueError(
                "Валидационный датасет пустой — вызовите _set_val_data перед fit"
            )

        self.model_ = self._build_model()

        calc_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

        generator = torch.Generator().manual_seed(self.random_state)

        train_data = TabularDataset(X, y)
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=generator,
        )

        best_loss = float("inf")
        # состояние модели сохраняется в чекпоинт, обновляется при лучшем лоссе
        # если в следующих эпохах модель станет хуже предиктить, можно будет вернуться к чекпоинту
        best_state = None
        epochs_without_improve = 0

        for epoch in range(self.epochs):
            # --- train ---
            self.model_.train()
            epoch_train_losses = []

            for batch_x, batch_y in tqdm(train_loader, leave=False):
                batch_x = batch_x.to(self.device_)
                # регрессия — float целевая переменная, форма (batch, 1)
                batch_y = batch_y.float().unsqueeze(1).to(self.device_)

                # forward
                pred = self.model_(batch_x)
                loss = calc_loss(pred, batch_y)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_losses.append(loss.item())

            mean_train_loss = np.mean(epoch_train_losses)
            train_rmse = np.sqrt(mean_train_loss)

            # --- validation ---
            self.model_.eval()
            epoch_val_losses = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device_)
                    batch_y = batch_y.float().unsqueeze(1).to(self.device_)

                    pred = self.model_(batch_x)
                    epoch_val_losses.append(calc_loss(pred, batch_y).item())

            mean_val_loss = np.mean(epoch_val_losses)
            # получаем RMSE из MSE
            val_rmse = np.sqrt(mean_val_loss)

            lr_scheduler.step(mean_val_loss)

            print(
                f"Epoch {epoch + 1}/{self.epochs} — "
                f"train RMSE: {train_rmse:.4f} | "
                f"val RMSE: {val_rmse:.4f}"
            )

            # --- early stopping по MSE, отображаем RMSE ---
            improvement_threshold = best_loss * self.best_loss_threshold_to_save

            if epoch == 0 or mean_val_loss < best_loss - improvement_threshold:
                epochs_without_improve = 0
                best_loss = mean_val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                print(f"  -> сохранён чекпоинт (val RMSE: {val_rmse:.4f})")
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= self.epochs_patience:
                print(f"Early stopping на эпохе {epoch + 1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X):
        if not self.model_:
            raise ValueError("Модель не обучена")

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(
                torch.tensor(np.array(X), dtype=torch.float32).to(self.device_)
            )
            # squeeze убирает размерность (batch, 1) -> (batch,) — как ожидает sklearn
            return logits.squeeze(1).cpu().numpy()

    # def score(self, X, y):
    #     """
    #     Возвращает negative RMSE — совместимо с get_scorer("neg_root_mean_squared_error").
    #     Больше = лучше, как принято в sklearn.
    #     """
    #     preds = self.predict(X)
    #     rmse = np.sqrt(np.mean((preds - np.array(y)) ** 2))
    #     return -rmse

    def _set_val_data(self, X, y):
        self.val_dataset = TabularDataset(X, y)


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

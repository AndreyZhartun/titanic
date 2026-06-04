"""
Адаптер для DNN с методами fit и transform.
Обучает нейронную сеть, валидирует и сохраняет лучшую модель в файл
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# для адаптации под api sklearn
from sklearn.base import BaseEstimator, ClassifierMixin


# наследование нужно для того, чтобы класс мог использоваться методами sklearn
class DNNAdapter(BaseEstimator, ClassifierMixin):
    # название файла модели с лучшим лоссом
    best_loss_file_name = "model_best_loss"

    def __init__(
        self,
        *,
        # дефолтные параметры нужны для адаптации под sklearn, описания см. также в конфиге
        # параметры слоев
        # входной размер
        in_features: int = 12,
        # размеры скрытых слоев
        hidden_sizes: list[int] = [128, 64],
        # размеры выходов
        out_features: int = 2,
        # параметры сети
        # вероятность дропаута
        dropout_rate: float = 0.25,
        # параметры разбития данных
        batch_size: int = 16,
        # параметры обучения
        learning_rate: float = 0.001,
        epochs: int = 5,
        # терпение для досрочного прерывания обучения
        epochs_patience: int = 10,
        best_loss_threshold_to_save: float = 0.001,
        # общие параметры
        random_state: int = 42,
        save_dir="pytorch_models",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.random_state = random_state
        self.save_dir = save_dir

        sizes = [in_features, *hidden_sizes, out_features]
        sizes_len = len(sizes)

        # слои нейронки
        layers = []

        # пройтись по парам размеров, например [12, 128] -> [128, 64] -> [64, 2]
        for i, size in enumerate(sizes):
            if i == 0:
                continue

            # последний слой
            is_last = i == sizes_len - 1
            # размер предыдущего слоя
            prev_size = sizes[i - 1]

            if is_last:
                layers.append(nn.Linear(prev_size, size))
                break

            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.BatchNorm1d(size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.model = nn.Sequential(*layers)

        self.model.to(self.device)

        # другие параметры
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_patience = epochs_patience
        self.best_loss_threshold_to_save = best_loss_threshold_to_save
        self.learning_rate = learning_rate

        self.calc_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

        # датасет с данными валидации, они сохраняются там перед fit, потому что сплит происходит перед передачей данных в fit
        self.val_dataset = None

    def fit(self, X, y):
        if self.val_dataset is None:
            raise ValueError(
                "Валидационный датасет в fit пустой, используйте _set_val_data перед вызовом fit"
            )

        train_data = MyDataset(X, y)
        val_data = self.val_dataset

        generator = torch.Generator().manual_seed(self.random_state)

        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=generator,
        )

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        lr_list = []
        best_loss = None
        # кол-во эпох без улучшений лосса
        epochs_without_loss_improve = 0

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

                pred_classes = torch.argmax(torch.softmax(pred, dim=1), dim=1).float()

                true_answers += (pred_classes == batch_y.float()).sum().item()

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss.append(loss.item())
                mean_train_loss = sum(running_train_loss) / len(running_train_loss)

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

            if mean_val_loss < best_loss - best_loss * self.best_loss_threshold_to_save:
                # обнулить кол-во эпох подряд без улучшения
                epochs_without_loss_improve = 0
                # обновить лосс
                best_loss = mean_val_loss

                self._save_state()
                print(f"epoch {epoch}: saved model with loss {best_loss}")
            else:
                epochs_without_loss_improve += 1

            # в конце каждой эпохи выдавать промежуточные результаты
            print(
                f"Epoch #{epoch}/{self.epochs} ended: {mean_train_loss=:.4f}, {running_train_acc=:.4f}, {mean_val_loss=:.4f}, {running_val_acc=:.4f}"
            )

            # досрочно прервать обучение
            if epochs_without_loss_improve == self.epochs_patience:
                print(f"Breaking on epoch {epoch}; loaded best loss model from file")
                self._load_state()
                break

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def _set_val_data(self, X, y):
        self.val_dataset = MyDataset(X, y)

    def _save_state(self, file_name_postfix=""):
        """
        Сохранить состояние модели и обучения в файл
        """

        try:
            os.mkdir(self.save_dir)
        except FileExistsError:
            pass

        file_path = os.path.join(
            self.save_dir, f"{DNNAdapter.best_loss_file_name}_{file_name_postfix}.pth"
        )

        state = {
            "model": self.model.state_dict(),
            "calc_loss": self.calc_loss.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        torch.save(state, file_path)

    def _load_state(self, file_name_postfix=""):
        """
        Загрузить состояние модели и обучения из файла
        """

        file_path = os.path.join(
            self.save_dir, f"{DNNAdapter.best_loss_file_name}_{file_name_postfix}.pth"
        )

        loaded_state = torch.load(file_path)

        self.model.load_state_dict(loaded_state.get("model"))
        self.calc_loss.load_state_dict(loaded_state.get("calc_loss"))
        self.optimizer.load_state_dict(loaded_state.get("optimizer"))
        self.lr_scheduler.load_state_dict(loaded_state.get("lr_scheduler"))


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

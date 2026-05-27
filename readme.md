### Titanic ML Pipeline

https://www.kaggle.com/competitions/titanic/overview

#### Установка

```
git clone .../titanic
cd titanic
# py -m venv .myvenv
# .myvenv/Scripts/activate
pip install -e .
```
Зависимости см. в pyproject.toml

#### Запуск обучения и предсказаний

```
python main.py
```
Предсказания будут сохранены в /submissions

#### Ноутбуки

- EDA - в ноутбуке notebook_eda проведено базовое EDA
- main - в ноутбуке notebook_main проведены эксперименты с разными параметрами моделей
- openFE - в ноутбуке notebook_openfe проведена генерация фичей с использованием openFE (csv фичей сохранены в /data)
----
- timeline - в ноутбуке notebook_timeline записана краткая история работы и TO DO


#### Структура проекта

- pipeline - класс пайплайна
- config - дефолтный конфиг пайплайна
- models - реестр моделей
- preprocessing - реестр препроцессоров
- dnn - класс нейронной сети
- utils - утилиты
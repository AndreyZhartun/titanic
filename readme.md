### ML Pipeline for Titanic & House Prices

https://www.kaggle.com/competitions/titanic/overview

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

Этот проект содержит пайплайны для двух датасетов одновременно.

В base классе содержится абстрактная логика запуска обучения и предсказаний, а отдельные классы-наследники реализуют логику для конкретных датасетов

Титаник лучше смотреть первым, потому что пайплайн изначально создавался под него, а потом был выделен в абстракцию

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

###### Titanic
```
python main.py
```

###### House Prices
```
python main_regression.py
```

Предсказания будут сохранены в /submissions

#### Ноутбуки

###### Titanic
- EDA - в ноутбуке titanic_eda проведено базовое EDA
- main - в ноутбуке titanic_main проведены эксперименты с разными параметрами моделей
- openFE - в ноутбуке notebook_openfe проведена генерация фичей с использованием openFE (csv фичей сохранены в /data)

###### House Prices
- EDA - в ноутбуке houseprices_eda проведено базовое EDA
- main - в ноутбуке houseprices_main проведены эксперименты с разными параметрами моделей

#### Ноутбук работы и комментариев
- timeline - в ноутбуке notebook_timeline записана краткая история работы и TO DO


#### Структура проекта
```
src/ml_pipeline
```
###### base

- pipeline - абстрактный класс пайплайна, от которого можно наследоваться
- base_preprocessors - абстрактные препроцессоры для любого датасета
- utils - утилиты

###### classification (titanic)
- config - дефолтный конфиг пайплайна классификации титаника
- models - реестр моделей классификации
- titanic_preprocessing - реестр препроцессоров именно для датасета титаника
- pipeline - класс-наследник для классификации
- dnn - класс нейронной сети для титаника

###### regression (house prices)
- config - дефолтный конфиг пайплайна регрессии цен на дома
- models - реестр моделей регрессии цен на дома
- preprocessing - реестр препроцессоров именно для датасета house prices
- pipeline - класс-наследник для регрессии цен на дома
- houseprices_dnn - класс нейронной сети для регрессии цен на дома
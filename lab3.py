# М8О-114СВ-24
# Чистяков Денис

# Лабораторная работа №3
# Оптимизация гиперпараметра

# Запуск контейнера:
# docker run --name postgres-optuna -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15.5


import optuna
from optuna.visualization import (
    plot_edf,
    plot_contour,
    plot_slice,
    plot_intermediate_values,
)
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def optimize_hyperparams(trial):
    # Загрузка датасета
    data = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Гиперпараметры
    n_trees = trial.suggest_int("n_trees", 50, 150)
    tree_depth = trial.suggest_int("tree_depth", 3, 30, step=3)
    min_split = trial.suggest_int("min_split", 2, 10)

    # Классификация
    model = RandomForestClassifier(
        n_estimators=n_trees, max_depth=tree_depth, min_samples_split=min_split, random_state=42
    )
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    # Оценка точности
    return precision_score(labels_test, predictions, average="macro")

# Подключение к хранилищу и начало исследования
storage_connection = "postgresql://postgres:example@localhost:5432/postgres"
exp_name = "iris_rf_optimization"

study = optuna.create_study(
    study_name=exp_name, direction="maximize", storage=storage_connection, load_if_exists=True
)
study.optimize(optimize_hyperparams, n_trials=30)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение:", study.best_value)

# Графики экспериментов
plot_edf(study).show()
plot_contour(study).show()
plot_slice(study).show()
plot_intermediate_values(study).show()

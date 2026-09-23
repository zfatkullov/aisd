import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score


DATA_URL = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "species"


class DataAnalyzer:
    # анализ датасета: признаки, баланс классов, диапазоны

    def __init__(self, df, target_col=TARGET_COL):
        self.df = df
        self.target_col = target_col

    def show_features(self):
        # все колонки кроме целевой
        features = [c for c in self.df.columns if c != self.target_col]
        print("Признаки объекта:", features)
        return features

    def class_balance(self):
        # сколько объектов в каждом классе
        balance = self.df[self.target_col].value_counts()
        print("\nБаланс классов (кол-во объектов на класс):")
        print(balance)
        return balance

    def describe_by_class(self):
        # средние по каждому классу
        print("\nСредние значения признаков по классам:")
        print(self.df.groupby(self.target_col)[FEATURE_COLS].mean())

    def min_max_by_class(self):
        # отсюда брал пороги для ручных правил
        print("\nДиапазоны (min-max) признаков по классам:")
        stats = self.df.groupby(self.target_col)[FEATURE_COLS].agg(["min", "max"])
        print(stats.to_string())  # чтобы не резало таблицу
        return stats


class ManualRuleClassifier:
    # классификация по порогам вручную

    @staticmethod
    def predict_one(row):
        pl = row["petal_length"]
        pw = row["petal_width"]
        if pl < 2.5:
            return "setosa"
        elif pl < 4.95 and pw < 1.7:
            return "versicolor"
        else:
            return "virginica"

    def predict(self, df):
        # прогоняем правило по всем строкам
        return df.apply(self.predict_one, axis=1)


class DecisionTreeExperiment:
    # обучаем дерево с разной глубиной и сравниваем

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []  # сюда копим результат по каждой глубине

    def run(self, depths=(3, 5, 7, 9)):
        for depth in depths:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(self.x_train, self.y_train)

            pred = tree.predict(self.x_test)
            acc = accuracy_score(self.y_test, pred)
            prec = precision_score(self.y_test, pred, average="macro", zero_division=0)

            self.results.append({
                "depth": depth,
                "accuracy": acc,
                "precision": prec,
                "model": tree,  # чтоб не переобучать заново
            })
            print(f"Глубина={depth}: accuracy={acc:.3f}, precision={prec:.3f}")
        return self.results

    def best_model(self):
        # берем лучшую по accuracy
        best = max(self.results, key=lambda r: r["accuracy"])
        return best["model"], best["depth"]


class LogisticRegressionGD:

    def __init__(self, n_features, n_classes, lr=0.5):
        self.lr = lr
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes))  # веса с нуля
        self.bias = np.zeros(n_classes)
        self.history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "w_norm": [],
            "w_update_norm": [],
        }

    @staticmethod
    def _softmax(z):
        z -= np.max(z, axis=1, keepdims=True)  # чтобы exp не улетал в бесконечность
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # в сумме дает 1

    def _one_hot(self, y):
        # 0/1/2 -> [1,0,0]/[0,1,0]/[0,0,1]
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, x, y, n_iterations):
        y_oh = self._one_hot(y)
        n = x.shape[0]

        for _ in range(n_iterations):
            weights_prev = self.weights.copy()  # запомнили веса до шага

            logits = x @ self.weights + self.bias
            probs = self._softmax(logits)

            loss = -np.mean(np.sum(y_oh * np.log(probs + 1e-12), axis=1))  # cross-entropy

            grad_w = x.T @ (probs - y_oh) / n
            grad_b = np.mean(probs - y_oh, axis=0)
            self.weights -= self.lr * grad_w  # шаг градиентного спуска
            self.bias -= self.lr * grad_b

            pred = np.argmax(probs, axis=1)  # класс с макс вероятностью
            acc = accuracy_score(y, pred)
            prec = precision_score(y, pred, average="macro", zero_division=0)

            # пишем в историю за эту итерацию
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
            self.history["precision"].append(prec)
            self.history["w_norm"].append(np.linalg.norm(self.weights))
            self.history["w_update_norm"].append(np.linalg.norm(self.weights - weights_prev))

        return self

    def predict(self, x):
        probs = self._softmax(x @ self.weights + self.bias)
        return np.argmax(probs, axis=1)


def plot_training_history(history, title_prefix, filename):
    # 4 графика: loss, accuracy/precision, норма весов, норма изменения весов
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["loss"], color="tab:red")
    axes[0, 0].set_title(f"{title_prefix}: потери (loss)")
    axes[0, 0].set_xlabel("итерация")
    axes[0, 0].set_ylabel("loss")

    axes[0, 1].plot(history["accuracy"], label="accuracy")
    axes[0, 1].plot(history["precision"], label="precision")
    axes[0, 1].set_title(f"{title_prefix}: accuracy / precision")
    axes[0, 1].set_xlabel("итерация")
    axes[0, 1].legend()

    axes[1, 0].plot(history["w_norm"], color="tab:green")
    axes[1, 0].set_title(f"{title_prefix}: L2-норма весов ||W||\u2082")
    axes[1, 0].set_xlabel("итерация")

    axes[1, 1].plot(history["w_update_norm"], color="tab:purple")
    axes[1, 1].set_title(f"{title_prefix}: L2-норма изменения весов ||\u0394W||\u2082")
    axes[1, 1].set_xlabel("итерация")

    plt.tight_layout()
    plt.savefig(filename)  # сохраняем в файл, не показываем на экране
    plt.close(fig)
    print(f"График сохранён: {filename}")


def load_dataset():
    df = pd.read_csv(DATA_URL)
    # приводим имена колонок к единому виду
    df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]
    return df

def plot_manual_rules(df, filename="manual_rules.png"):
    setosa = df[df["species"] == "setosa"]
    versicolor = df[df["species"] == "versicolor"]
    virginica = df[df["species"] == "virginica"]
    plt.figure(figsize=(20, 10))

    plt.scatter(setosa["petal_length"], setosa["petal_width"], color="blue", label="setosa")
    plt.scatter(versicolor["petal_length"], versicolor["petal_width"], color="orange", label="versicolor")
    plt.scatter(virginica["petal_length"], virginica["petal_width"], color="green", label="virginica")

    wrong = df[df["species"] != df["manual_prediction"]]
    plt.scatter(wrong["petal_length"], wrong["petal_width"], facecolors="none", edgecolors="red", s=150, label="ошибка")

    plt.axvline(x=2.5, color="gray", linestyle="--")
    plt.axvline(x=4.8, color="black", linestyle="--")
    plt.axhline(y=1.7, color="purple", linestyle="--")

    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print("График сохранён:", filename)

def run_manual_rules(df):
    print("\n=== Ручные правила ===")
    rule_clf = ManualRuleClassifier()
    df["manual_prediction"] = rule_clf.predict(df)
    acc = accuracy_score(df[TARGET_COL], df["manual_prediction"])
    print(f"Точность ручных правил: {acc:.3f}")

    out_cols = FEATURE_COLS + [TARGET_COL, "manual_prediction"]
    df[out_cols].to_csv("manual_rules_predictions.csv", index=False)
    plot_manual_rules(df)
    print("Сохранено: manual_rules_predictions.csv")


def run_decision_tree(df, x_train, x_test, y_train, y_test, x_all, label_encoder):
    print("\n=== Дерево решений ===")
    experiment = DecisionTreeExperiment(x_train, x_test, y_train, y_test)
    results = experiment.run(depths=(3, 5, 7, 9))
    best_tree, best_depth = experiment.best_model()
    print(f"Лучшая глубина по accuracy: {best_depth}")

    metrics_df = pd.DataFrame([
        {"depth": r["depth"], "accuracy": r["accuracy"], "precision": r["precision"]}
        for r in results
    ])
    metrics_df.to_csv("decision_tree_metrics.csv", index=False)
    print("Сохранено: decision_tree_metrics.csv")

    # прогоняем лучшую модель на всем датасете, не только на тесте
    pred_full = best_tree.predict(x_all)
    tree_out = df[FEATURE_COLS + [TARGET_COL]].copy()
    tree_out["tree_prediction"] = label_encoder.inverse_transform(pred_full)  # числа обратно в названия
    tree_out.to_csv("decision_tree_predictions.csv", index=False)
    print("Сохранено: decision_tree_predictions.csv")


def run_logistic_regression(df, x_train_s, x_test_s, x_all_s, y_train, y_test, label_encoder):
    print("\n=== Линейная регрессия ===")
    iteration_values = [10, 30, 50, 70, 100]
    metrics = []
    final_model = None

    for n_iter in iteration_values:
        # каждый раз обучаем новую модель с нуля на своем числе итераций
        model = LogisticRegressionGD(
            n_features=x_train_s.shape[1],
            n_classes=len(label_encoder.classes_),
            lr=2,
        )
        model.fit(x_train_s, y_train, n_iterations=n_iter)

        pred_test = model.predict(x_test_s)
        acc = accuracy_score(y_test, pred_test)
        prec = precision_score(y_test, pred_test, average="macro", zero_division=0)
        metrics.append({"n_iterations": n_iter, "accuracy": acc, "precision": prec})
        print(f"Итераций={n_iter}: accuracy={acc:.3f}, precision={prec:.3f}")

        final_model = model  # в конце останется модель со 100 итерациями

    pd.DataFrame(metrics).to_csv("linear_regression_metrics.csv", index=False)
    print("Сохранено: linear_regression_metrics.csv")

    pred_full = final_model.predict(x_all_s)
    lr_out = df[FEATURE_COLS + [TARGET_COL]].copy()
    lr_out["lr_prediction"] = label_encoder.inverse_transform(pred_full)
    lr_out.to_csv("linear_regression_predictions.csv", index=False)
    print("Сохранено: linear_regression_predictions.csv")

    plot_training_history(
        final_model.history,
        "Линейная регрессия (100 итераций)",
        "linear_regression_history.png",
    )


class RobustnessTester:
    # прогоняем логрегрессию на разных train/test сплитах + меряем время и память

    def __init__(self, x, y, n_classes):
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self.rows = []

    @staticmethod
    def _measure(func, *args, **kwargs):
        # обертка, чтобы не дублировать замер времени/памяти отдельно
        tracemalloc.start()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, elapsed, peak / 1024  # переводим байты в кб

    # ДОБАВЛЕНО: train_size=0.6 (разбиение 60/40) — попросил преподаватель
    # проверить, что accuracy не падает и на этом разбиении в том числе
    def run(self, random_states=(0, 1, 42), train_sizes=(0.5, 0.6, 0.7, 0.9)):
        # 3 сида х 4 размера выборки = 12 комбинаций
        for rs in random_states:
            for train_size in train_sizes:
                x_train, x_test, y_train, y_test = train_test_split(
                    self.x, self.y, train_size=train_size,
                    random_state=rs, stratify=self.y,
                )
                scaler = StandardScaler()
                x_train_s = scaler.fit_transform(x_train)
                x_test_s = scaler.transform(x_test)

                def fit_logreg():
                    model = LogisticRegressionGD(
                        n_features=x_train_s.shape[1],
                        n_classes=self.n_classes,
                        lr=2,
                    )
                    model.fit(x_train_s, y_train, n_iterations=100)
                    return model

                logreg, l_time, l_mem = self._measure(fit_logreg)
                logreg_acc = accuracy_score(y_test, logreg.predict(x_test_s))

                self.rows.append({
                    "random_state": rs,
                    "train_size": train_size,
                    # ДОБАВЛЕНО: читаемая подпись разбиения (60/40 и т.д.) для таблицы/графика в отчёте
                    "split_label": f"{int(train_size * 100)}/{100 - int(train_size * 100)}",
                    "n_test_samples": len(y_test),
                    "logreg_accuracy": round(logreg_acc, 3),
                    "logreg_time_sec": round(l_time, 5),
                    "logreg_peak_mem_kb": round(l_mem, 2),
                })
                print(f"rs={rs}, train_size={train_size}: "
                      f"logreg_acc={logreg_acc:.3f} ({l_time*1000:.2f} мс)")
        return pd.DataFrame(self.rows)


def run_robustness_tests(x, y, n_classes):
    print("\n=== Тестирование логрегрессии на разных train/test разбиениях ===")
    tester = RobustnessTester(x, y, n_classes)
    results_df = tester.run()
    results_df.to_csv("robustness_test_results.csv", index=False)
    print("Сохранено: robustness_test_results.csv")

    # ДОБАВЛЕНО: сводка + график, чтобы наглядно показать устойчивость accuracy
    # к выбору разбиения (нужно для отчёта и для ответа преподавателю)
    plot_robustness_summary(results_df)
    return results_df


# ДОБАВЛЕНО: строит сводную таблицу (среднее accuracy по random_state)
# и график accuracy логрегрессии по разбиениям - сохраняется в robustness_accuracy.png
def plot_robustness_summary(results_df, filename="robustness_accuracy.png"):
    order = [f"{int(ts * 100)}/{100 - int(ts * 100)}" for ts in sorted(results_df["train_size"].unique())]
    summary = results_df.groupby("split_label").agg(
        logreg_acc_mean=("logreg_accuracy", "mean"),
    ).reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(summary))
    ax.plot(x_pos, summary["logreg_acc_mean"], marker="s", color="tab:orange", label="Регрессия")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary.index)
    ax.set_xlabel("Разбиение train/test")
    ax.set_ylabel("Accuracy (среднее по random_state)")
    ax.set_title("Устойчивость accuracy логрегрессии к разным разбиениям train/test")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close(fig)
    print(f"График сохранён: {filename}")
    print("\nСводная таблица (среднее по random_state):")
    print(summary.round(4).to_string())
    return summary


def main():
    df = load_dataset()
    print(df.head())

    analyzer = DataAnalyzer(df)
    analyzer.show_features()
    analyzer.class_balance()
    analyzer.describe_by_class()
    analyzer.min_max_by_class()

    run_manual_rules(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])
    x = df[FEATURE_COLS].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    run_decision_tree(df, x_train, x_test, y_train, y_test, x, label_encoder)

    # для логрегрессии добавляем новый признак - площадь лепестка,
    # он даёт нелинейную комбинацию исходных данных и поднимает accuracy
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    x_lr = df[FEATURE_COLS + ["petal_area"]].values

    x_train_lr, x_test_lr, y_train, y_test = train_test_split(
        x_lr, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_lr)
    x_test_s = scaler.transform(x_test_lr)
    x_all_s = scaler.transform(x_lr)

    run_logistic_regression(df, x_train_s, x_test_s, x_all_s, y_train, y_test, label_encoder)

    run_robustness_tests(x, y, n_classes=len(label_encoder.classes_))


if __name__ == "__main__":
    main()
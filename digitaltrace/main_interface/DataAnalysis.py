import re
import numpy as np
import sklearn
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

class DataAnalysis:

    @staticmethod
    def train_by_model(dataframe_model, column_for_learning: str, model_training_method: str, test_size: float,
                   n_estimators: int, max_depth: int, min_samples_split: int, min_samples_leaf: int, max_features: int):
        """Обучение по модели

        Args:
            dataframe_model (dataframe): dataframe с обработанной моделью

            column_for_learning (str): столбец, значения которого будут предсказываться

            model_training_method (str): тип модели для обучения

            test_size (float): размер выборки, который будет предназначен для тестирования (0.1 - 0.9)

            n_estimators (int): сколько деревьев решений необходимо создать в процессе обучения модели

            max_depth (int): максимальная глубина дерева решений

            min_samples_split (int): минимальное количество образцов (наблюдений), которые необходимо иметь в узле, чтобы он мог быть разделен на две ветви. 

            min_samples_leaf (int): минимальное количество объектов, которое должно быть в листовой вершине дерева решений. 

            max_features (int): максимальное количество признаков, которые рассматриваются при поиске наилучшего разделения на каждом узле дерева решений. 

        Returns:
            clf: обученная модель
            classification_quality_assessment: оценка качества обученной аналитической модели
            X: все столбцы кроме column_for_learning для входных данных
            y_test: данные прогнозируемого столбца для тестирования
            y_pred: предсказания по обученной модели
            results_df: dataframe для вывода результатов
        """

        training_models = {'RandomForestRegressor': RandomForestRegressor, 'GradientBoostingRegressor': GradientBoostingRegressor,
                        'RandomForestClassifier': RandomForestClassifier, 'GradientBoostingClassifier': GradientBoostingClassifier}

        # Получить тип обучающей модели (Classifier/Regressor)
        def get_last_word(model):
            model_name = model.__name__
            last_word = re.findall('[A-Z][^A-Z\\s]*$', model_name)[-1]
            return last_word

        type_of_classification = get_last_word(
            training_models[model_training_method])
        
        # Поиск межквартильного размаха
        Q1 = dataframe_model[column_for_learning].quantile(0.25)
        Q3 = dataframe_model[column_for_learning].quantile(0.75)
        IQR = Q3 - Q1

        # Определение границ выбросов
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 0.5 * IQR

        # Удаление выбросов
        dataframe_model = dataframe_model[(dataframe_model[column_for_learning] >= lower_bound) & (dataframe_model[column_for_learning] <= upper_bound)]

        # сохранить все столбцы кроме column_for_learning для входных данных
        X = dataframe_model.drop(column_for_learning, axis=1)
        # задать column_for_learning как столбец выходных данных (прогнозируемый)
        y = dataframe_model[column_for_learning]
        y = y.astype('int')
        # разбиение на данные для обучения и тестирования
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=42)

        # Вычисление весов примеров
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # обучение модели
        clf = training_models[model_training_method](n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        clf.fit(X_train, y_train, sample_weight=sample_weights)

        # сделать предсказания по обученной модели
        y_pred = clf.predict(X_test)
        # округление значений
        y_pred = np.round(y_pred).astype(int)
        
        # Создание DataFrame для вывода результатов
        results_df = pd.DataFrame(X_test, columns=X.columns)
        results_df[column_for_learning] = y_test
        results_df['Predicted'] = y_pred

        if type_of_classification == 'Classifier':
            # оценки классификации precision, recall, fscore
            acc = sklearn.metrics.accuracy_score(y_test, y_pred)
            precision_score = sklearn.metrics.precision_score(
                y_test, y_pred, average='macro', zero_division=0)
            recall_score = sklearn.metrics.recall_score(
                y_test, y_pred, average='macro')
            f1_score = sklearn.metrics.f1_score(y_test, y_pred, average='macro')

            classification_quality_assessment = (
                f"Accuracy оценка: {acc:.2f}\n" +
                f"Precision оценка: {precision_score:.2f}\n" +
                f"Recall оценка: {recall_score:.2f}\n" +
                f"F1 оценка: {f1_score:.2f}\n"
            )

            return clf, classification_quality_assessment, X, y_test, y_pred, results_df

        else:
            # оценки для регрессии MSE, RMSE, MAE, R^2
            mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
            r2 = sklearn.metrics.r2_score(y_test, y_pred)

            regression_quality_assessment = (
                f"Среднеквадратичная ошибка предсказания (MSE):{mse:.2f}\n" +
                f"Квадратный корень из среднеквадратичной ошибки (RMSE):{rmse:.2f}\n" +
                f"Средняя абсолютная ошибка (MAE):{mae:.2f}\n" +
                f"Коэффициент детерминации (R^2):{r2:.2f}"
            )

            return clf, regression_quality_assessment, X, y_test, y_pred, results_df
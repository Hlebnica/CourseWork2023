import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

class GraphsBuilder:
    
    @staticmethod
    def feature_importance_graph(clf, X, figsize=(12,6)):
        """График важности признаков

        Args:
            clf: обученная модель
            X (dataframe): все столбцы кроме column_for_learning для входных данных из DataAnalysis
            figsize (int, int): размер графика. По умолчанию (12,6).

        Returns:
            feature_importance_graph: график важности признаков
        """
        
        
        y_axis_labels = {
            'CheckType': 'Тип контроля',
            'CourseEduId' : 'Id в EduSusu',
            'CourseNumber' : 'Номер курса',
            'DirectionCode' : 'Код специальности',
            'DirectionName': 'Название специальности',
            'Id': 'Id в "Универис"',
            'IsPractice': 'Является ли практикой',
            'Speciality': 'Квалификация',
            'StudyForm': 'Форма обучения',
            'SubjectName': 'Название предмета',
            'Term': 'Семестр',
            'Year': 'Год',
            'EnrollScore': 'Вступительные баллы',
            'FinancialForm': 'Форма финансирования\nобучения',
            'HasOlympParticipation': 'Участие в олипиаде',
            'LiveCity': 'Город проживания',
            'RegisterCity': 'Город регистрации',
            'Sex': 'Пол',
            'Status': 'Статус обучения',
            'Mark': 'Полученная оценка',
            'Rating': 'Рейтинг по БРС',
            'StudentId': 'Id студента\nв “Универис”',
            'journal_id': 'Id записи',
            'student_id': 'Id студента',
            'Mark1': 'Баллы 1 предмет',
            'Mark2': 'Баллы 2 предмет',
            'Mark3': 'Баллы 3 предмет',
            'Subject1': '1 предмет',
            'Subject2': '2 предмет',
            'Subject3': '3 предмет', 
            'FinancialForm_budget_share_decile': 'Квантиль бюджетников\nв группе',
            'EgeMark1': '1 дециль\nоценки за ЕГЭ',
            'EgeMark2': '2 дециль\nоценки за ЕГЭ',
            'EgeMark3': '3 дециль\nоценки за ЕГЭ',
            'MarkSecondSem ': 'Полученная оценка\nза 2 семестр',
            'DecileSumRatingStudent1Sem': 'Дециль суммарного рейтинга\nза 1 семестр',
            'DecileSumRatingStudent2Sem': 'Дециль суммарного рейтинга\nза 2 семестр',
            'DecileMedianRatingCredits1Sem': 'Дециль медианного рейтинга\nпо зачетам за 1 семестр',
            'DecileMedianRatingCredits2Sem': 'Дециль медианного рейтинга\nпо зачетам за 2 семестр',
            'DecileMedianRatingExam1Sem': ' Дециль медианного рейтинга\nпо экзаменам за 1 семестр',
            'DecileMedianRatingExam2Sem': ' Дециль медианного рейтинга\nпо экзаменам за 2 семестр',
            'MarkFirstSem': 'Дециль оценки\nза 1 семестр',
            'MarkSecondSem': 'Дециль оценки\nза 2 семестр',
            'Predicted': 'Предсказанный результат',
        }
        
        importance = clf.feature_importances_
        feat_importances = pd.Series(importance, index=X.columns)
        
        fig = plt.figure(figsize=figsize) 
        ax = fig.add_subplot(111)
        feat_importances.nlargest(10).plot(kind='barh', ax=ax) 
        
        # Переименовывание столбцов
        ax.set_yticklabels([y_axis_labels.get(label, label) for label in feat_importances.nlargest(10).index])
        
        # Сохранение графика в файл
        plt.savefig('feature_importance_graph.png')
        
        # Преобразование графика в кодировку base64
        fig = plt.gcf()
        plt.subplots_adjust(left=0.2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        feature_importance_graph = base64.b64encode(image_png)
        feature_importance_graph = feature_importance_graph.decode('utf-8')
        buffer.close()
    
        return feature_importance_graph
    
    @staticmethod
    def plot_predictions(y_test, y_pred, figsize=(12,6)):
        """График предсказаний и реальных значений

        Args:
            y_test: данные прогнозируемого столбца для тестирования из DataAnalysis
            y_pred: предсказания по обученной модели из DataAnalysis
            figsize (int, int): размер графика. По умолчанию (12,6).

        Returns:
            plot_predictions_graph: график предсказаний и реальных значений
        """
        
        fig = plt.figure(figsize=figsize)
    
        # Построение гистограммы
        plt.hist(y_test.values, alpha=0.5, label='Истинные значения', color='blue')
        plt.hist(y_pred, alpha=0.5, label='Предсказанные значения', color='orange')
        
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.subplots_adjust(left=0.2, right=0.8)
        
        # Сохранение графика в файл
        plt.savefig('plot_predictions_hist.png')
        
        # Преобразование графика в кодировку base64
        fig = plt.gcf()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        plot_predictions_hist = base64.b64encode(image_png)
        plot_predictions_hist = plot_predictions_hist.decode('utf-8')
        buffer.close()
        
        return plot_predictions_hist
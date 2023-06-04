import asyncio
import pandas as pd
import json
from .AsyncRequests import AsyncRequests
from .DataClear import DataClear
from .TransformationsOverDataframe import TransformationsOverDataframe
from .PredictiveModels import PredictiveModels
from .DataAnalysis import DataAnalysis
from .GraphsBuilder import GraphsBuilder 
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import urllib
import os
import re

from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.core.cache import cache
from django.views import View
from django.contrib import messages


dfByYear = None
dfStudentsByJournalId = None
dfRatingByJournalId = None
dfEgeMarksByStudentsId = None


def df_to_html(df):
    """Преобразование dataframe в html

    Args:
        df (dataframe): dataframe, который необходимо перевести в html

    Returns:
        html: dataframe преобразованный в html
    """
    
    column_names_dict = {
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
        'FinancialForm': 'Форма финансирования обучения',
        'HasOlympParticipation': 'Участие в олипиаде',
        'LiveCity': 'Город проживания',
        'RegisterCity': 'Город регистрации',
        'Sex': 'Пол',
        'Status': 'Статус обучения',
        'Mark': 'Полученная оценка',
        'Rating': 'Рейтинг по БРС',
        'StudentId': 'Id студента в “Универис”',
        'journal_id': 'Id записи',
        'student_id': 'Id студента',
        'Mark1': 'Баллы 1 предмет',
        'Mark2': 'Баллы 2 предмет',
        'Mark3': 'Баллы 3 предмет',
        'Subject1': '1 предмет',
        'Subject2': '2 предмет',
        'Subject3': '3 предмет', 
        'FinancialForm_budget_share_decile': 'Квантиль бюджетников в группе',
        'EgeMark1': '1 дециль оценки за ЕГЭ',
        'EgeMark2': '2 дециль оценки за ЕГЭ',
        'EgeMark3': '3 дециль оценки за ЕГЭ',
        'MarkSecondSem ': 'Полученная оценка за 2 семестр',
        'DecileSumRatingStudent1Sem': 'Дециль суммарного рейтинга за 1 семестр',
        'DecileSumRatingStudent2Sem': 'Дециль суммарного рейтинга за 2 семестр',
        'DecileMedianRatingCredits1Sem': 'Дециль медианного рейтинга по зачетам за 1 семестр',
        'DecileMedianRatingCredits2Sem': 'Дециль медианного рейтинга по зачетам за 2 семестр',
        'DecileMedianRatingExam1Sem': ' Дециль медианного рейтинга по экзаменам за 1 семестр',
        'DecileMedianRatingExam2Sem': ' Дециль медианного рейтинга по экзаменам за 2 семестр',
        'MarkFirstSem': 'Дециль оценки за 1 семестр',
        'MarkSecondSem': 'Дециль оценки за 2 семестр',
        'Predicted': 'Предсказанный результат',
    }
    df_renamed = df.rename(columns=column_names_dict)
    df_html = df_renamed.to_html(classes='table table-bordered scrollable', index=False)
    return df_html

 
def predictive_model_result(predictive_model, academic_subject, type_of_control, 
                            dfByYear, dfStudentsByJournalId, dfRatingByJournalId, dfEgeMarksByStudentsId):
    """Формирование обучающей выборки в зависимости от модели

    Args:
        predictive_model (string): название модели данных
        academic_subject (string): предмет по которому будет идти анализ
        type_of_control (string): тип контроля предмета по которому будет идти анализ
        dfByYear (dataframe): dataframe с журналом по годам
        dfStudentsByJournalId (dataframe): dataframe с инфомарцией по студентам
        dfRatingByJournalId (dataframe): dataframe с рейтингом студентов
        dfEgeMarksByStudentsId (dataframe): dataframe с ЕГЭ и вступительными студентов

    Returns:
        semestPerfomance: dataframe с обучающей выборкой
    """
    
    type_of_control_inner = ''
    if type_of_control == 'credit':
        type_of_control_inner = 'Зачет'
    else:
        type_of_control_inner = 'Экзамен'
    if predictive_model == 'model1' and academic_subject and dfByYear:
        modelDisciplineFirstSemestr = PredictiveModels.model_discipline_number_semestr(academic_subject, 1, dfByYear[0], dfStudentsByJournalId, 
                                                                                       dfRatingByJournalId, dfEgeMarksByStudentsId)
        return modelDisciplineFirstSemestr
    if predictive_model == 'model2' and academic_subject and dfByYear:
        modelDisciplineSecondSemestr = PredictiveModels.model_discipline_second_semestr(academic_subject, dfByYear[0],dfStudentsByJournalId, 
                                                                                        dfRatingByJournalId, dfEgeMarksByStudentsId)
        return modelDisciplineSecondSemestr
    if predictive_model == 'model3' and type_of_control and dfByYear:
        semestPerfomance = PredictiveModels.model_semester_performance(dfByYear[0], dfStudentsByJournalId, 
                                                                       dfRatingByJournalId, dfEgeMarksByStudentsId, 
                                                                       1, 1, type_of_control_inner, True)
        return semestPerfomance
    if predictive_model == 'model4' and type_of_control and dfByYear:
        semestPerfomance = PredictiveModels.model_semester_performance(dfByYear[0], dfStudentsByJournalId, 
                                                                       dfRatingByJournalId, dfEgeMarksByStudentsId, 
                                                                       1, 2, type_of_control_inner, True)
        return semestPerfomance
    if predictive_model == 'model5' and type_of_control and dfByYear:
        semestPerfomance = PredictiveModels.model_semester_performance(dfByYear[0], dfStudentsByJournalId, 
                                                                       dfRatingByJournalId, dfEgeMarksByStudentsId, 
                                                                       1, 1, type_of_control_inner, False)
        return semestPerfomance
    if predictive_model == 'model6' and type_of_control and dfByYear:
        semestPerfomance = PredictiveModels.model_semester_performance(dfByYear[0], dfStudentsByJournalId, 
                                                                       dfRatingByJournalId, dfEgeMarksByStudentsId, 
                                                                       1, 2, type_of_control_inner, False)
        return semestPerfomance
    return None


def analysis_model_result(predictive_model, type_of_control, df_predictive_model, model_training_method, test_size, n_estimators, 
                                      max_depth, min_samples_split, min_samples_leaf, max_features):
    
        """Формирование прогнозной модели
        
        Args:
            predictive_model (string): название модели данных
        
            type_of_control (string): тип контроля предмета
        
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
            analysis_model: обученная модель
        """   
        column_for_learning = ''
        if predictive_model == 'model1':
            column_for_learning = 'Mark'
        elif predictive_model == 'model2':
            column_for_learning =  'MarkSecondSem'
        elif predictive_model == 'model3':
            column_for_learning =  'Status'
        elif predictive_model == 'model4':
            column_for_learning =  'Status'      
        elif predictive_model == 'model5':
            if type_of_control == 'credit':
                column_for_learning = 'DecileMedianRatingCredits1Sem'
            else:
                column_for_learning = 'DecileMedianRatingExam1Sem'
        elif predictive_model == 'model6':
            if type_of_control == 'credit':
                column_for_learning = 'DecileMedianRatingCredits2Sem'
            else:
                column_for_learning = 'DecileMedianRatingExam2Sem'

        analysis_model = DataAnalysis.train_by_model(dataframe_model=df_predictive_model, column_for_learning=column_for_learning, 
                                                    model_training_method=model_training_method, test_size=float(test_size),
                                                    n_estimators=int(n_estimators), max_depth=int(max_depth), 
                                                    min_samples_split = int(min_samples_split), 
                                                    min_samples_leaf=int(min_samples_leaf), 
                                                    max_features=int(max_features))

        return analysis_model

# index
def get_data_api(request): 
    """Генерация интерфейса взаимодействия со страницей экстракции данных из API

    Args:
        request: объект запроса

    Returns:
        html: интерфейс взаимодействия со страницей экстракции данных из API
    """
    
    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    
    institutes_list = ["Высшая школа электроники и компьютерных наук", "Архитектурно-строительный институт", "Высшая медико-биологическая школа", "Высшая школа экономики и управления", 
                        "Институт естественных и точных наук", "Институт лингвистики и международных коммуникаций", "Институт медиа и социально-гуманитарных наук", 
                        "Институт спорта, туризма, сервиса", "Политехнический институт", "Юридический институт", "Филиалы ЮУрГУ"]
    
    if request.method == 'GET' and 'page' in request.GET:
        
        institutes_list_item = request.GET.get('institutes_list') 
         
        institutes_copy = institutes_list.copy()  
        institutes_copy.remove(institutes_list_item)  # Удалить строку с ВШ, которую нужно оставить
         
        flat_excluded_institutes = DataClear.institutional_exclusion(institutes_copy)
        year_from = request.GET.get('year_from') 
        year_to = request.GET.get('year_to') 
        range_from = request.GET.get('range_from') 
        range_to = request.GET.get('range_to') 
        high_schools = request.GET.get('high_schools')
        
        years_list = []
        for i in range(int(year_from), int(year_to) + 1):
            years_list.append(i)
        
        course_range = [1, 2, 3, 4, 5, 6]
        filtered_course_range = [x for x in course_range if x < int(range_from) or x > int(range_to)]

        dfByYear = asyncio.run(AsyncRequests.journal_by_years(years_list)) 
        dfByYear[0] = DataClear.drop_rows_in_journal(dfByYear[0], CheckType=['курсовые работы', 
                                                                                'практика', 
                                                                                'курсовые проекты', 
                                                                                'дипломный проект'],
                        Speciality=['магистр', 'специалист', 'аспирант'],
                        StudyForm=['заочная', 'очно-заочная'],
                        CourseNumber=filtered_course_range,
                        DirectionCode=flat_excluded_institutes)

        # Запрос студентов по id за N год
        dfStudentsByJournalId = asyncio.run(AsyncRequests.students_by_journal_id(dfByYear[0]))

        # Запрос оценок за предмет по id за N год
        dfRatingByJournalId = asyncio.run(AsyncRequests.rating_by_journal_id(dfByYear[0]))

        # Получение результатов ЕГЭ по StudentsByJournalId
        dfEgeMarksByStudentsId = asyncio.run(AsyncRequests.ege_marks_by_student_id(dfStudentsByJournalId))
        dfEgeMarksByStudentsId = TransformationsOverDataframe.ege_marks_transpose(dfEgeMarksByStudentsId)
        
        # Перевод файлов в csv для скачивания
        dfByYear_csv = dfByYear[0].to_csv(index=False)
        dfStudentsByJournalId_csv = dfStudentsByJournalId.to_csv(index=False)
        dfRatingByJournalId_csv = dfRatingByJournalId.to_csv(index=False)
        dfEgeMarksByStudentsId_csv = dfEgeMarksByStudentsId.to_csv(index=False)

        # Перевод в html
        df_year_now_html = df_to_html(dfByYear[0].head(100))
        df_students_by_journal_id_html = df_to_html(dfStudentsByJournalId.head(100))
        df_rating_by_journal_id_html = df_to_html(dfRatingByJournalId.head(100))
        df_ege_by_students_id_html = df_to_html(dfEgeMarksByStudentsId.head(100))

        return render(request, 'index.html', {
            'year_from': year_from,
            'year_to': year_to,
            'range_from': range_from,
            'range_to': range_to,
            'high_schools': high_schools,
            
            'df_year_now_html': df_year_now_html,
            'df_students_by_journal_id_html': df_students_by_journal_id_html,
            'df_rating_by_journal_id_html': df_rating_by_journal_id_html,
            'df_ege_by_students_id_html': df_ege_by_students_id_html,
            
            'df_year_size': dfByYear[0].shape[0], 
            'df_students_by_journal_size': dfStudentsByJournalId.shape[0],
            'df_rating_by_journal_size': dfRatingByJournalId.shape[0],
            'df_ege_by_students_size': dfEgeMarksByStudentsId.shape[0],
            
            'dfByYear_csv': dfByYear_csv,
            'dfStudentsByJournalId_csv': dfStudentsByJournalId_csv, 
            'dfRatingByJournalId_csv': dfRatingByJournalId_csv,
            'dfEgeMarksByStudentsId_csv': dfEgeMarksByStudentsId_csv,
            
            'institutes_list': institutes_list,
            'institutes_list_item': institutes_list_item,
        })
    else:
        return render(request, 'index.html', {'institutes_list': institutes_list})
    
    
    
def get_info_from_csv(request):
    """Генерация интерфейса взаимодействия со страницей экстракции данных из CSV файлов

    Args:
        request: объект запроса

    Returns:
        html: интерфейс взаимодействия со страницей экстракции данных из CSV файлов
    """

    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    
    if request.method == 'POST' and 'page' in request.POST:
        
        file_dfYear = request.FILES['file_dfYear']
        file_dfStudents = request.FILES['file_dfStudents']
        file_dfRating = request.FILES['file_dfRating']
        file_dfEge = request.FILES['file_dfEge']
        
        dfByYear = [1]
        dfByYear[0] = pd.read_csv(file_dfYear)
        dfStudentsByJournalId = pd.read_csv(file_dfStudents)
        dfRatingByJournalId = pd.read_csv(file_dfRating)
        dfEgeMarksByStudentsId = pd.read_csv(file_dfEge)
        
        year = re.findall(r'\d{4}-\d{4}', file_dfYear.name)  # поиск всех цифр в названии файла по шаблону
        
        df_year_now_html = df_to_html(dfByYear[0].head(100))
        df_students_by_journal_id_html = df_to_html(dfStudentsByJournalId.head(100))
        df_rating_by_journal_id_html = df_to_html(dfRatingByJournalId.head(100))
        df_ege_by_students_id_html = df_to_html(dfEgeMarksByStudentsId.head(100))
        
        return render(request, 'import-csv.html', {
            'df_year_now_html': df_year_now_html,
            'df_students_by_journal_id_html': df_students_by_journal_id_html,
            'df_rating_by_journal_id_html': df_rating_by_journal_id_html,
            'df_ege_by_students_id_html': df_ege_by_students_id_html,
            'year': year[0],
            
            'df_year_size': dfByYear[0].shape[0], 
            'df_students_by_journal_size': dfStudentsByJournalId.shape[0],
            'df_rating_by_journal_size': dfRatingByJournalId.shape[0],
            'df_ege_by_students_size': dfEgeMarksByStudentsId.shape[0],
        })
    
    return render(request, 'import-csv.html')
    
    
    
# analysis
def get_data_analysis(request):
    """Генерация интерфейса взаимодействия со страницей Анализа данных

    Args:
        request: объект запроса

    Returns:
        html: интерфейс взаимодействия со страницей Анализа данных
    """
    
    global dfByYear 
    global dfStudentsByJournalId
    global dfRatingByJournalId
    global dfEgeMarksByStudentsId
    
    df_predictive_model = None
    
    if request.method == 'GET' and 'page' in request.GET:
        try:
            predictive_model = request.GET.get('predictive_model')
            type_of_control = request.GET.get('type_of_control')  

            subject_name_1_term = request.GET.get('subject_name_1_term')  
            subject_name_1_2_term = request.GET.get('subject_name_1_2_term')  
            
            subject_name_1_term_item = subject_name_1_term
            subject_name_1_2_term_item = subject_name_1_2_term

            if predictive_model == 'model1':
                academic_subject = subject_name_1_term
            else:
                academic_subject = subject_name_1_2_term 

            model_training_method = request.GET.get('model_training_method')  
            test_size = request.GET.get('test_size')  
            n_estimators = request.GET.get('n_estimators')  
            max_depth = request.GET.get('max_depth')  
            min_samples_split = request.GET.get('min_samples_split')  
            min_samples_leaf = request.GET.get('min_samples_leaf')  
            max_features = request.GET.get('max_features')  
            
            # Предметы, которые были в 1 семестре
            subject_name_1_term = dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 1).dropna().unique().tolist()
            
            # Предметы, которые были в 1 и продолжаются во 2 семестре
            term_1_subjects = set(dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 1).unique())
            term_2_subjects = set(dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 2).unique())
            subject_name_1_2_term = list(term_1_subjects.intersection(term_2_subjects))
            
            # Формирование обучающей выборки
            df_predictive_model = predictive_model_result(predictive_model, academic_subject, type_of_control, 
                                                        dfByYear, dfStudentsByJournalId, dfRatingByJournalId, dfEgeMarksByStudentsId)
            
            # Анализа данных
            analysis_model = analysis_model_result(predictive_model, type_of_control, df_predictive_model, model_training_method,
                                                float(test_size), int(n_estimators), int(max_depth),
                                                int(min_samples_split), int(min_samples_leaf), int(max_features)) 
            
            df_predictive_model_html = df_to_html(df_predictive_model)
            
            # Графики
            feature_importance_graph = GraphsBuilder.feature_importance_graph(analysis_model[0], analysis_model[2])
            predictions_graph = GraphsBuilder.plot_predictions(analysis_model[3], analysis_model[4])
        
            return render(request, 'analysis.html', {
                'predictive_model': predictive_model,
                'academic_subject': academic_subject,
                'type_of_control': type_of_control,
                'df_predictive_model_html': df_predictive_model_html,
                'prediction_result': analysis_model[1],
                'feature_importance_graph': feature_importance_graph,
                'predictions_graph': predictions_graph,
                'results_df': df_to_html(analysis_model[5]),
                
                'model_training_method': model_training_method,
                'test_size': test_size,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features,

                'df_predictive_model_size': df_predictive_model.shape[0],
                
                'subject_name_1_term': subject_name_1_term,
                'subject_name_1_2_term': subject_name_1_2_term,

                'subject_name_1_term_item': subject_name_1_term_item,
                'subject_name_1_2_term_item': subject_name_1_2_term_item,
            })
        except TypeError:
            messages.warning(request, 'Отсутствуют данные для обучения')
        except ValueError:
            messages.warning(request, 'Проблема с выборкой данных или гиперпараметрами')
        
    if dfByYear:
        # Предметы, которые были в 1 семестре
        subject_name_1_term = dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 1).dropna().unique().tolist()
        
        # Предметы, которые были в 1 и продолжаются во 2 семестре
        term_1_subjects = set(dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 1).unique())
        term_2_subjects = set(dfByYear[0]['SubjectName'].where(dfByYear[0]['Term'] == 2).unique())
        subject_name_1_2_term = list(term_1_subjects.intersection(term_2_subjects))
        
        return render(request, 'analysis.html', {
            'subject_name_1_term': subject_name_1_term,
            'subject_name_1_2_term': subject_name_1_2_term,
        })
     
    return render(request, 'analysis.html')


def get_info(requset):
    """Генерация интерфейса информации о программе

    Args:
        request: объект запроса

    Returns:
        html: интерфейс с информацией о программе
    """
    
    return render(requset, 'info.html')
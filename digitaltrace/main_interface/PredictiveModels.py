from .DataClear import DataClear
from .ModelsSimplification import ModelsSimplification
import pandas as pd
import numpy as np


class PredictiveModels:
    @staticmethod
    def model_discipline_number_semestr(subject_name: str, term_number: int, journal_by_year_dataframe, students_by_journal_id_dataframe,
                                        rating_by_journal_id_dataframe, ege_marks_by_students_id_dataframe):
        """Дисциплина {subject_name} за {term_number} семестр

        Args:
            subjectName(str): название предмета 
            term_number(int): номер семестра предмета
            journal_by_year_dataframe(dataframe): dataframe с журналом по году
            students_by_journal_id_dataframe(dataframe): dataframe с журналом по студентам
            rating_by_journal_id_dataframe(dataframe): dataframe с журналом по рейтингу по предмету
            ege_marks_by_students_id_dataframe(dataframe): dataframe с журналом оценок ЕГЭ студентов

        Returns:
            merged_df: dataframe с новым столбцом

        """

        # ---- jouranlByYearDataframe -------------------------------------------------------------------------------

        # замена типа контроля на экзамен, дифференцированный зачет - 1 // зачет - 0
        journal_by_year_dataframe['CheckType'] = journal_by_year_dataframe['CheckType'].replace(
            {'дифференцированный зачет': 1, 'экзамен': 1, 'зачет': 0})

        # начальное формирование модели из журнала за год
        journal_by_year_dataframe = DataClear.keep_select_columns(journal_by_year_dataframe, 'CheckType',
                                                                  'CourseNumber',
                                                                  'Id', 'SubjectName', 'Term')
        # оставить только выбранный предмет(subjectName) в номере курса(CourseNumber)
        journal_by_year_dataframe = DataClear.keep_rows_in_journal(journal_by_year_dataframe, CourseNumber=[1],
                                                                   SubjectName=[subject_name])
        # ------------------------------------------------------------------------------------------------------------

        # ---- studentsByJournalIdDataframe --------------------------------------------------------------------------
        # замена формы финансроивания стундента на бюджет - 1 // контракт - 0
        students_by_journal_id_dataframe['FinancialForm'] = students_by_journal_id_dataframe['FinancialForm'].replace(
            {'бюджет': 1, 'контракт': 0})

        # замена пола стундента на Мужской - 1 // Женский - 0
        students_by_journal_id_dataframe['Sex'] = students_by_journal_id_dataframe['Sex'].replace(
            {'Мужской': 1, 'Женский': 0})

        # замена города регистрации местный - г.Челябинск - 1 // иначе - 0
        students_by_journal_id_dataframe['RegisterCity'] = students_by_journal_id_dataframe['RegisterCity'].apply(
            lambda x: 1 if isinstance(x, str) and x.replace(" ", "") == "г.Челябинск" or x == 1 else 0)

        # дециль доля бюджетников группа
        budget_share_decile = students_by_journal_id_dataframe.groupby(
            'journal_id')['FinancialForm'].quantile(0.001)
        students_by_journal_id_dataframe = students_by_journal_id_dataframe.merge(budget_share_decile, on='journal_id',
                                                                                  suffixes=['', '_budget_share_decile'])

        students_by_journal_id_dataframe = DataClear.keep_select_columns(students_by_journal_id_dataframe,
                                                                         'RegisterCity',
                                                                         'FinancialForm', 'Sex',
                                                                         'FinancialForm_budget_share_decile', 'Id',
                                                                         'journal_id')
        # -------------------------------------------------------------------------------------------------------------

        # ---- ratingByJournalIdDataframe -----------------------------------------------------------------------------
        rating_by_journal_id_dataframe = DataClear.keep_select_columns(rating_by_journal_id_dataframe, 'Mark',
                                                                       'journal_id')
        # -------------------------------------------------------------------------------------------------------------

        # ---- egeMarksByStudentsIdDataframe -------------------------------------------------------------------------
        ege_marks_by_students_id_dataframe = DataClear.keep_select_columns(ege_marks_by_students_id_dataframe,
                                                                           'student_id',
                                                                           'Mark1', 'Mark2', 'Mark3')

        ege_marks_by_students_id_dataframe[['Mark1', 'Mark2', 'Mark3']] = ege_marks_by_students_id_dataframe[
            ['Mark1', 'Mark2', 'Mark3']].apply(lambda x: pd.qcut(x, 5, labels=False) + 1)

        # ------------------------------------------------------------------------------------------------------------------

        # Объединение таблиц
        merged_df = pd.merge(journal_by_year_dataframe, students_by_journal_id_dataframe, left_on='Id',
                             right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, rating_by_journal_id_dataframe, left_on='journal_id', right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, ege_marks_by_students_id_dataframe, left_on='Id_y', right_on='student_id',
                             how='inner')

        # Сохранить term_number семестр
        merged_df = DataClear.keep_rows_in_journal(
            merged_df, Term=[term_number])

        # Убрать дубликаты
        merged_df.drop_duplicates(
            subset='student_id', keep='first', inplace=True)

        merged_df = merged_df.rename(
            columns={'Mark1': 'EgeMark1', 'Mark2': 'EgeMark2', 'Mark3': 'EgeMark3'})

        merged_df = DataClear.keep_select_columns(merged_df, 'Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity',
                                                  'FinancialForm', 'FinancialForm_budget_share_decile', 'Mark')

        return merged_df

    @staticmethod
    def model_discipline_second_semestr(subject_name: str, journal_by_year_dataframe, students_by_journal_id_dataframe,
                                        rating_by_journal_id_dataframe, ege_marks_by_students_id_dataframe):
        """Дисциплина 2 семестр с таким же предметом в прошлом семестре

        Args:
            subjectName(str): название предмета 2 семестра, который был в 1 семестре
            journal_by_year_dataframe(dataframe): dataframe с журналом по году
            students_by_journal_id_dataframe(dataframe): dataframe с журналом по студентам
            rating_by_journal_id_dataframe(dataframe): dataframe с журналом по рейтингу по предмету
            ege_marks_by_students_id_dataframe(dataframe): dataframe с журналом оценок ЕГЭ студентов

        Returns:
            merged_df: dataframe с новыми столбцами

        """

        # ---- jouranlByYearDataframe ---------------------------------------------------------------------------------
        # замена типа контроля на экзамен, дифференцированный зачет - 1 // зачет - 0
        journal_by_year_dataframe['CheckType'] = journal_by_year_dataframe['CheckType'].replace(
            {'дифференцированный зачет': 1, 'экзамен': 1, 'зачет': 0})

        # начальное формирование модели из журнала за год
        journal_by_year_dataframe = DataClear.keep_select_columns(journal_by_year_dataframe, 'CheckType',
                                                                  'CourseNumber',
                                                                  'Id', 'SubjectName', 'Term')
        # оставить только выбранный предмет(subjectName) в номере курса(CourseNumber)
        journal_by_year_dataframe = DataClear.keep_rows_in_journal(journal_by_year_dataframe, CourseNumber=[1],
                                                                   SubjectName=[subject_name])
        # -------------------------------------------------------------------------------------------------------------

        # ---- studentsByJournalIdDataframe ---------------------------------------------------------------------------
        # замена формы финансроивания стундента на бюджет - 1 // контракт - 0
        students_by_journal_id_dataframe['FinancialForm'] = students_by_journal_id_dataframe['FinancialForm'].replace(
            {'бюджет': 1, 'контракт': 0})

        # замена пола стундента на Мужской - 1 // Женский - 0
        students_by_journal_id_dataframe['Sex'] = students_by_journal_id_dataframe['Sex'].replace(
            {'Мужской': 1, 'Женский': 0})

        # замена города регистрации местный - г.Челябинск - 1 // иначе - 0
        students_by_journal_id_dataframe['RegisterCity'] = students_by_journal_id_dataframe['RegisterCity'].apply(
            lambda x: 1 if isinstance(x, str) and x.replace(" ", "") == "г.Челябинск" or x == 1 else 0)

        # дециль доля бюджетников группа
        budget_share_decile = students_by_journal_id_dataframe.groupby(
            'journal_id')['FinancialForm'].quantile(0.001)
        students_by_journal_id_dataframe = students_by_journal_id_dataframe.merge(budget_share_decile, on='journal_id',
                                                                                  suffixes=['', '_budget_share_decile'])

        students_by_journal_id_dataframe = DataClear.keep_select_columns(students_by_journal_id_dataframe,
                                                                         'RegisterCity',
                                                                         'FinancialForm', 'Sex',
                                                                         'FinancialForm_budget_share_decile', 'Id',
                                                                         'journal_id')
        # -----------------------------------------------------------------------------------------------------------

        # ---- ratingByJournalIdDataframe --------------------------------------------------------------------------
        rating_by_journal_id_dataframe = DataClear.keep_select_columns(rating_by_journal_id_dataframe, 'Mark',
                                                                       'journal_id')
        # -----------------------------------------------------------------------------------------------------------

        # ---- egeMarksByStudentsIdDataframe -------------------------------------------------------------------------
        ege_marks_by_students_id_dataframe = DataClear.keep_select_columns(ege_marks_by_students_id_dataframe,
                                                                           'student_id',
                                                                           'Mark1', 'Mark2', 'Mark3')

        ege_marks_by_students_id_dataframe[['Mark1', 'Mark2', 'Mark3']] = ege_marks_by_students_id_dataframe[
            ['Mark1', 'Mark2', 'Mark3']].apply(lambda x: pd.qcut(x, 5, labels=False) + 1)

        # ----------------------------------------------------------------------------------------------------------

        # Объединение таблиц
        merged_df = pd.merge(journal_by_year_dataframe, students_by_journal_id_dataframe, left_on='Id',
                             right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, rating_by_journal_id_dataframe, left_on='journal_id', right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, ege_marks_by_students_id_dataframe, left_on='Id_y', right_on='student_id',
                             how='inner')

        # Убрать дубликаты
        merged_df.drop_duplicates(
            subset=["student_id", "Term"], keep="first", inplace=True)

        # создание нового столбца MarkSecondSem на основе столбца Mark и Term
        merged_df['MarkSecondSem'] = merged_df.apply(
            lambda row: row['Mark'] if row['Term'] == 1 else np.nan, axis=1)

        # заполнение значений в столбце MarkSecondSem на основе столбца Mark и Term
        merged_df['MarkSecondSem'] = merged_df['MarkSecondSem'].fillna(
            merged_df['Mark'].where(merged_df['Term'] == 2))

        merged_df = merged_df.rename(
            columns={'Mark1': 'EgeMark1', 'Mark2': 'EgeMark2', 'Mark3': 'EgeMark3', 'Mark': 'MarkFirstSem'})

        merged_df = DataClear.keep_select_columns(merged_df, 'Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity',
                                                  'FinancialForm', 'FinancialForm_budget_share_decile', 'CheckType',
                                                  'MarkFirstSem', 'MarkSecondSem')

        return merged_df

    @staticmethod
    def model_semester_performance(journal_by_year_dataframe, students_by_journal_id_dataframe, rating_by_journal_id_dataframe, ege_marks_by_students_id_dataframe,
                                   term_number_start: int, term_number_end: int, check_type: str, forecast_type: bool):
        """Успеваемость по семестру

         Args:
            journal_by_year_dataframe(dataframe): dataframe с журналом по году
            students_by_journal_id_dataframe(dataframe): dataframe с журналом по студентам
            rating_by_journal_id_dataframe(dataframe): dataframe с журналом по рейтингу по предмету
            ege_marks_by_students_id_dataframe(dataframe): dataframe с журналом оценок ЕГЭ студентов
            term_number_start(int): от какого семестра брать выборку
            term_number_end(int): до какого семестра брать выборку
            check_type(int): зачет/экзамен 
            forecast_type(bool): добавить столбец Status в итоговую таблицу со значениями Учится - 1 // Отчислен - 0

        Returns:
            merged_df: dataframe с новым столбцом 
        """

        if check_type.lower() == 'зачет':
            check_type = 0
        elif check_type.lower() == 'экзамен':
            check_type = 1

        # ---- jouranlByYearDataframe ----------------------------------------------------------------------------------------------------
        # замена типа контроля на экзамен, дифференцированный зачет - 1 // зачет - 0
        journal_by_year_dataframe['CheckType'] = journal_by_year_dataframe['CheckType'].replace(
            {'дифференцированный зачет': 1, 'экзамен': 1, 'зачет': 0})

        # начальное формирование модели из журнала за год
        journal_by_year_dataframe = DataClear.keep_select_columns(journal_by_year_dataframe, 'CheckType', 'CourseNumber',
                                                                  'Id', 'SubjectName', 'Term')

        # Список семестров
        terms = []
        for i in range(term_number_start, term_number_end + 1):
            terms.append(i)

        # оставить только выбранный Номер курса(CourseNumber) и Семестр (Term)
        journal_by_year_dataframe = DataClear.keep_rows_in_journal(
            journal_by_year_dataframe, CourseNumber=[1], Term=terms)
        # --------------------------------------------------------------------------------------------------------------------------------

        # ---- studentsByJournalIdDataframe --------------------------------------------------------------------------
        # замена формы финансроивания стундента на бюджет - 1 // контракт - 0
        students_by_journal_id_dataframe['FinancialForm'] = students_by_journal_id_dataframe['FinancialForm'].replace(
            {'бюджет': 1, 'контракт': 0})

        # замена пола стундента на Мужской - 1 // Женский - 0
        students_by_journal_id_dataframe['Sex'] = students_by_journal_id_dataframe['Sex'].replace(
            {'Мужской': 1, 'Женский': 0})

        # замена статуса обучения стундента на Учится - 1 // Отчислен - 0
        students_by_journal_id_dataframe['Status'] = students_by_journal_id_dataframe['Status'].replace(
            {'учится': 1, 'отчислен': 0})

        # замена города регистрации местный - г.Челябинск - 1 // иначе - 0
        students_by_journal_id_dataframe['RegisterCity'] = students_by_journal_id_dataframe['RegisterCity'].apply(
            lambda x: 1 if isinstance(x, str) and x.replace(" ", "") == "г.Челябинск" or x == 1 else 0)

    # дециль доля бюджетников группа
        budget_share_decile = students_by_journal_id_dataframe.groupby(
            'journal_id')['FinancialForm'].quantile(0.001)
        students_by_journal_id_dataframe = students_by_journal_id_dataframe.merge(budget_share_decile, on='journal_id',
                                                                                  suffixes=['', '_budget_share_decile'])

        # Убрать людей в академе
        students_by_journal_id_dataframe = DataClear.drop_rows_in_journal(
            students_by_journal_id_dataframe, Status=['в академе'])

        students_by_journal_id_dataframe = DataClear.keep_select_columns(students_by_journal_id_dataframe,
                                                                         'RegisterCity',
                                                                         'FinancialForm', 'Sex',
                                                                         'FinancialForm_budget_share_decile', 'Id',
                                                                         'journal_id', 'Status')
        # -------------------------------------------------------------------------------------------------------------

        # ---- ratingByJournalIdDataframe -----------------------------------------------------------------------------
        rating_by_journal_id_dataframe = DataClear.keep_select_columns(rating_by_journal_id_dataframe, 'Rating',
                                                                       'journal_id')
        # -------------------------------------------------------------------------------------------------------------

        # ---- egeMarksByStudentsIdDataframe -------------------------------------------------------------------------
        ege_marks_by_students_id_dataframe = DataClear.keep_select_columns(ege_marks_by_students_id_dataframe,
                                                                           'student_id',
                                                                           'Mark1', 'Mark2', 'Mark3')

        ege_marks_by_students_id_dataframe[['Mark1', 'Mark2', 'Mark3']] = ege_marks_by_students_id_dataframe[
            ['Mark1', 'Mark2', 'Mark3']].apply(lambda x: pd.qcut(x, 5, labels=False) + 1)
        # ----------------------------------------------------------------------------------------------------------

        # Объединение таблиц
        merged_df = pd.merge(journal_by_year_dataframe, students_by_journal_id_dataframe, left_on='Id',
                             right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, rating_by_journal_id_dataframe, left_on='journal_id', right_on='journal_id',
                             how='inner')
        merged_df = pd.merge(merged_df, ege_marks_by_students_id_dataframe, left_on='Id_y', right_on='student_id',
                             how='inner')

        # Убрать дубликаты
        merged_df.drop_duplicates(
            subset=["student_id", "Term"], keep="first", inplace=True)

        # Переименовывание столбцов результатов ЕГЭ
        merged_df = merged_df.rename(
            columns={'Mark1': 'EgeMark1', 'Mark2': 'EgeMark2', 'Mark3': 'EgeMark3'})

        merged_df = DataClear.keep_select_columns(merged_df, 'Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity',
                                                  'FinancialForm', 'FinancialForm_budget_share_decile', 'CheckType',
                                                  'Status', 'student_id', 'journal_id', 'Term', 'Rating'
                                                  )

        columns_for_keep = ['Sex', 'EgeMark1', 'EgeMark2', 'EgeMark3', 'RegisterCity',
                            'FinancialForm', 'FinancialForm_budget_share_decile', ]

        # ДецильВгруппеСуммаРейтинговСтудентаПоВсемДисциплинам{term_number}Семестр
        for n in range(term_number_start, term_number_end + 1):
            merged_df = ModelsSimplification.decile_group_sum_rating_students_by_semestr(
                merged_df, n)
            columns_for_keep.append(f'DecileSumRatingStudent{n}Sem')

        if check_type == 0:
            # Убрать все экзамены
            merged_df = DataClear.drop_rows_in_journal(
                merged_df, CheckType=[1])
        else:
            # Убрать все зачеты
            merged_df = DataClear.drop_rows_in_journal(
                merged_df, CheckType=[0])

        # ДецильМедианныйРейтингДисциплин{check_type}{term_number}Семестр
        for m in range(term_number_start, term_number_end + 1):
            merged_df = ModelsSimplification.decile_median_rating_disciplin_check_type_by_semestr(
                merged_df, check_type, m)
            if check_type == 0:
                columns_for_keep.append(f'DecileMedianRatingCredits{m}Sem')
            else:
                columns_for_keep.append(f'DecileMedianRatingExam{m}Sem')

        # Если True то добавить в конец статус обучения
        if forecast_type:
            columns_for_keep.append('Status')

        merged_df = DataClear.keep_select_columns_list(
            merged_df, columns_for_keep)

        return merged_df

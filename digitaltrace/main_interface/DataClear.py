import pandas as pd


class DataClear:
    
    # Высшие школы и их коды
    __INSTITUTES = {
        "Архитектурно-строительный институт": ["07.03.01", "07.03.03", "08.03.01", "21.03.02", "54.03.01", "08.05.01"],
        "Высшая медико-биологическая школа": ["19.03.03", "37.03.01", "37.05.01", "37.05.02"],
        "Высшая школа экономики и управления": ["09.03.02", "09.03.03", "38.03.01", "38.03.01", "38.03.02", "38.03.03",
                                                "38.03.04", "38.03.05", "38.05.01", "38.05.02"],
        "Высшая школа электроники и компьютерных наук": ["02.03.02", "02.03.02", "09.03.01", "09.03.04", "10.03.01",
                                                         "11.03.02", "11.03.03", "12.03.01", "27.03.04", "10.05.03",
                                                         "11.05.01", "24.05.06"],
        "Институт естественных и точных наук": ["01.03.02", "01.03.03", "01.03.04", "02.03.01", "03.03.01", "04.03.01",
                                                "05.03.06", "11.03.04", "18.03.01", "18.03.02"],
        "Институт лингвистики и международных коммуникаций": ["41.03.01", "41.03.04", "41.03.05", "45.03.01",
                                                              "45.03.02",
                                                              "45.03.03", "45.05.01"],
        "Институт медиа и социально-гуманитарных наук": ["39.03.01", "42.03.01", "42.03.02", "45.03.01", "46.03.01"],
        "Институт спорта, туризма, сервиса": ["19.03.04", "29.03.04", "43.03.02", "43.03.03", "44.03.01", "49.03.01"],
        "Политехнический институт": ["13.03.01", "13.03.02", "13.03.03", "15.03.01", "15.03.01", "15.03.02", "15.03.03",
                                     "15.03.04", "15.03.05", "15.03.06", "15.03.06", "20.03.01", "22.03.01", "22.03.02",
                                     "23.03.01", "23.03.02", "23.03.03", "24.03.01", "24.03.04", "17.05.01", "20.05.01",
                                     "23.05.01", "23.05.02", "24.05.01", "24.05.02"],
        "Юридический институт": ["40.03.01", "40.05.01", "40.05.02", "40.05.03", "40.05.03", "40.05.04"],
        "Филиалы ЮУрГУ": ["05.03.01", "19.03.02", "43.03.01", "27.03.02"]
    }

    @staticmethod
    def institutional_exclusion(institutes):
        """Список для исключения высших школ из выборки

        Args:
            institutes (List): высшие школы, которые будут исключены

        Returns:
            flat_excluded_institutes: список высших школ
        """
        
        institutes_list = institutes
        excluded_institutes = [value for key, value in DataClear.__INSTITUTES.items() if key in institutes_list]
        flat_excluded_institutes = [item for sublist in excluded_institutes for item in sublist]
        return flat_excluded_institutes

    @staticmethod
    def drop_rows_in_journal(df, **kwargs):
        """Удаление строк в журнале dataframe по словарям

        Args:
            df (dataframe): dataframe
            
            **kwargs: название поля из журнала=[значения для удаления]

        Returns:
            df: dataframe с удаленными строками
        """

        for key, values in kwargs.items():
            df = df[~df[key].isin(values)]
        return df

    @staticmethod
    def keep_rows_in_journal(df, **kwargs):
        """Сохранение выбранных строк в журнале dataframe по словарям

        Args:
            df (dataframe): dataframe
            
            **kwargs: название поля из журнала=[значения, которые будут сохранены]

        Returns:
            df: dataframe с сохраненными строками
        """

        for key, values in kwargs.items():
            df = df[df[key].isin(values)]
        return df

    @staticmethod
    def drop_select_columns(df, *columns):
        """Удаляет указанные столбцы из DataFrame

        Args:
            df (dataframe): dataframe
            
            *columns: колонки для удаления

        Returns:
            dataframe: dataframe с удаленными колонками
        """
        return df.drop(columns=columns)

    @staticmethod
    def keep_select_columns(df, *columns):
        """Сохраняет указанные столбцы из DataFrame

        Args:
            df (dataframe): dataframe
            
            *columns: колонки, которые будут сохранены

        Returns:
            dataframe: dataframe с сохраненными колонками
        """
        return df.loc[:, columns]
    

    @staticmethod
    def keep_select_columns_list(df, columns):
        """Сохраняет указанные столбцы из DataFrame

        Args:
            df (dataframe): dataframe
            
            columns (List): колонки, которые будут сохранены

        Returns:
            dataframe: dataframe с сохраненными колонками
        """
        return df.loc[:, columns]

import aiohttp
import json


class DigitalTrace:
    __BASE_URL = ""

    @staticmethod
    async def get_journal_by_year(year):
        """Получить журнал по году

        Args:
            year (int): год, по которому нужно получить журнал

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetJournals/Year/{year}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_teachers_by_journal_id(id_subject):
        """Получить преподавателей по id из журнала по годам

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetTeachersByJournalId/{id_subject}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_students_by_journal_id(id_subject):
        """Получить студентов по id из журнала по годам

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetStudentsByJournalId/{id_subject}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_grades_by_journal_id(id_subject):
        """Задания предметов студетов по id журнала

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetGradesByJournalId/{id_subject}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_rating_by_journal_id(id_subject):
        """Оценки за предметы по id журнала

        Args:
            id_subject (string): id из журнала по годам

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetRatingsByJournalId/{id_subject}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url, ssl=False) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

    @staticmethod
    async def get_ege_marks_by_student_id(id_subject):
        """Оценки ЕГЭ студентов по id студентов

        Args:
            id_subject (string): id из журнала по студентам

        Returns:
            json: результат запроса
        """
        
        url = DigitalTrace.__BASE_URL + f"GetEgeMarksByStudentId/{id_subject}"
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                data = await response.json()
                return json.dumps(data, ensure_ascii=False, indent=4)

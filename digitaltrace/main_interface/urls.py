from django.urls import include, path
from main_interface import views
# from .views import YearDataView

urlpatterns = [
     path('', views.get_data_api, name='index'),
     path('analysis/', views.get_data_analysis, name='analysis'),
     path('import-csv/', views.get_info_from_csv, name='import-csv'),
     path('info/', views.get_info, name='info'),
]
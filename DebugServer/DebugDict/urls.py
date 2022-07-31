from django.urls import path
from . import views


app_name = 'DebugDict'
urlpatterns = [
    path('', views.SearchView.as_view(), name='index'),
]

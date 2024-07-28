# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('spain_form/', views.spain_form, name='spain_form'),
    path('future/', views.future, name='future'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('story/', views.story, name='story'),
]

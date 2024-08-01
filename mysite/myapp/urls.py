# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('spain_form/', views.spain_form, name='spain_form'),
    path('contact/', views.contact, name='contact'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('story/', views.story, name='story'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('user_spain_form/', views.user_spain_form, name='user_spain_form')
]

# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('spain_form/', views.spain_form, name='spain_form'),
    path('contact/', views.contact, name='contact'),
    path('portfolio/', views.portfolio, name='portfolio'),
    path('story/', views.story, name='story'),
    path('login/', views.user_login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logoutUser, name='logout'),
    path('user_spain_form/', views.user_spain_form, name='user_spain_form'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('user_dashboard_spain/', views.user_dashboard_spain, name='user_dashboard_spain'),
    path('profile/', views.profile, name='profile'),
    path('change_password/', views.change_password, name='change_password'),
]

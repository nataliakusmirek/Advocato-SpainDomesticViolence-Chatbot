# myapp/urls.py

from django.urls import path
from . import views
from django.conf.urls.i18n import i18n_patterns 

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
    path('set_language/', views.set_language, name='set_language'),
    path('profile/', views.profile, name='profile'),
    path('change_password/', views.change_password, name='change_password'),
    path('search/', views.search_view, name='search'),
]

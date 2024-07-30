from django.shortcuts import render
from django.http import HttpResponse
from django.core.mail import send_mail
from django.conf import settings
from django.shortcuts import redirect

# Create your views here.
def index(request):
    return render(request, 'index.html')

def spain_form(request):
    return render(request, 'spain_form.html')

def contact(request):
    return render(request, 'contact.html')

def portfolio(request):
    return render(request, 'portfolio.html')

def story(request):
    return render(request, 'story.html')

def login(request):
    return render(request, 'login.html')

def register(request):
    return render(request, 'register.html')
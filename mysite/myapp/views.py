from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')

def spain_form(request):
    return render(request, 'spain_form.html')

def future(request):
    return render(request, 'future.html')

def portfolio(request):
    return render(request, 'portfolio.html')

def story(request):
    return render(request, 'story.html')

def login(request):
    return render(request, 'login.html')

def register(request):
    return render(request, 'register.html')
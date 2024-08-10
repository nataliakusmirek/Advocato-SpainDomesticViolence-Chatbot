import csv
import os
import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.mail import send_mail
from django.conf import settings
from django.shortcuts import redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from .forms import SpainForm
from django.contrib.auth.decorators import login_required
from django.utils.translation import get_language
from django.utils import translation
import logging
import pandas as pd

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

def logoutUser(request):
    logout(request)
    return redirect('index')

def user_login(request):
    page = 'login'
    if request.user.is_authenticated:
        print('User is already authenticated')
        return redirect('user_dashboard')
    

    if request.method == 'POST':
        username = request.POST.get('username').lower()
        password = request.POST.get('password')

        try:
            user = User.objects.get(username=username)
        except:
            messages.error(request, 'User does not exist')
            return render(request, 'login.html', {'page': page})

        user = authenticate(request, username=username, password=password)
        if user is not None:
            print('User is authenticated, logging in...')
            login(request, user)
            return redirect('user_dashboard')
        else:
            messages.error(request, 'Username OR password does not exist.')
    context = {'page': page}
    return render(request, 'login.html', context)

def register(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('user_dashboard')
        else:
            messages.error(request, 'An error occurred during registration')
    
    return render(request, 'register.html', {'form': form})
   
def user_spain_form(request):
    if request.method == 'POST':
        form = SpainForm(request.POST)
        if form.is_valid():
            cleaned_data = form.cleaned_data
            risk_ranking = cleaned_data.get('risk_ranking')  # Use get() to safely retrieve values
            language = cleaned_data.get('language')

            # Set the language
            translation.activate(language)
            request.LANGUAGE_CODE = language
            
            try:
                save_form_data_to_csv(cleaned_data, risk_ranking, language)
                messages.success(request, 'Your form has been submitted successfully.')
                return redirect('index')
            except Exception as e:
                print(e)
                messages.error(request, 'An error occurred during form submission. Please try again.')
        else:
            print(form.errors)
            messages.error(request, 'There were errors in the form submission. Please try again.')
    else:
        form = SpainForm()    
    return render(request, 'user_spain_form.html', {'form': form})

def set_language(request):
    user_language = request.GET.get('language', 'en')
    if user_language:
        translation.activate(user_language)
        request.session['django_language'] = user_language
        messages.success(request, f'Language changed to {user_language}')
    return redirect(request.META.get('HTTP_REFERER', '/'))


def save_form_data_to_csv(data, risk_ranking):
    file_path = os.path.join(os.path.dirname(__file__), 'spain_form_responses', 'responses.csv')

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the header for the CSV file if it does not exist
    fieldnames = [
        'name', 'age', 'gender', 'nationality', 'contact_information', 
        'date_of_incident', 'location_of_incident', 'violence_type', 
        'description_of_incident', 'witnesses', 'witnesses_contact_information',
        'perpetrator_name', 'perpetrator_relationship', 'perpetrator_age',
        'perpetrator_gender', 'history_of_violence', 'previous_incidents',
        'previous_incidents_reported', 'action_taken', 'access_to_weapons',
        'threats_made', 'victim_afraid', 'children_dependents', 'safe_place',
        'medical_attention', 'safe_housing', 'legal_help', 'counseling',
        'other_support', 'additional_information', 'consent', 'risk_ranking', 'language'
    ]

    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            data['risk_ranking'] = risk_ranking
            data['language'] = get_language()
            writer.writerow(data)
    except Exception as e:
        messages.error(f'An error occured while saving the form data to the CSV file: {e}')

def user_dashboard(request):
    return render(request, 'user_dashboard.html')

def user_dashboard_spain(request):
    return render(request, 'user_dashboard_spain.html')


def profile(request):
    return render(request, 'profile.html')

def change_password(request):
    if request.method == 'POST':
        new_password = request.POST.get('password')
        if new_password:
            request.user.set_password(new_password)
            request.user.save()
            messages.success(request, 'Your password has been successfully updated.')
            return redirect('profile')  # Redirect to the profile page or any other page after success
        else:
            messages.error(request, 'Please enter a new password.')

    return render(request, 'profile.html')

def search_view(request):
    query = request.GET.get('query', '').lower()
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'myapp', 'spain_form_responses', 'responses.csv'))
    
    # Filter rows containing the query string in any column
    filtered_df = df.apply(lambda row: row.astype(str).str.contains(query).any(), axis=1)
    filtered_data = df[filtered_df].to_dict(orient='records')

    return JsonResponse(filtered_data, safe=False)
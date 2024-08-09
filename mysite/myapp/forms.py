from django import forms
from django.utils.translation import gettext_lazy as _

class SpainForm(forms.Form):
    # Victim Information
    name = forms.CharField(label=_('Name'), max_length=100, required=True)
    age = forms.IntegerField(label=_('Age'), required=True)
    gender = forms.ChoiceField(label=_('Gender'), choices=[('Female', 'Female'), ('Male', 'Male'), ('Other', 'Other')], required=True)
    nationality = forms.CharField(label=_('Nationality'), max_length=100, required=True)
    contact_information = forms.EmailField(label=_('Contact Information'), max_length=100, required=True)

    # Incident Details
    date_of_incident = forms.DateField(label=_('Date of Incident'), required=True)
    location_of_incident = forms.CharField(label=_('Location of Incident'), max_length=100, required=True)
    violence_type = forms.ChoiceField(label=_('Type of Violence Experienced'), choices=[
        ('Physical', 'Physical'),
        ('Emotional/Psychological', 'Emotional/Psychological'),
        ('Sexual', 'Sexual'),
        ('Financial/Economic', 'Financial/Economic'),
        ('Other', 'Other')
    ], required=True)
    description_of_incident = forms.CharField(label=_('Description of the Incident'), widget=forms.Textarea, required=True)
    witnesses = forms.CharField(label=_('Witnesses (if any)'), widget=forms.Textarea, required=False)
    witnesses_contact_information = forms.EmailField(label=_('Witnesses Contact Information'), max_length=100, required=False)

    # Perpetrator Information
    perpetrator_name = forms.CharField(label=_('Perpetrator Name'), max_length=100, required=True)
    perpetrator_relationship = forms.ChoiceField(label=_('Relationship to Victim'), choices=[
        ('Partner/Spouse', 'Partner/Spouse'),
        ('Ex-Partner/Ex-Spouse', 'Ex-Partner/Ex-Spouse'),
        ('Family Member', 'Family Member'),
        ('Friend/Acquaintance', 'Friend/Acquaintance'),
        ('Stranger', 'Stranger'),
        ('Other', 'Other')
    ], required=True)
    perpetrator_age = forms.IntegerField(label=_('Perpetrator Age'), required=True)
    perpetrator_gender = forms.ChoiceField(label=_('Perpetrator Gender'), choices=[('Female', 'Female'), ('Male', 'Male'), ('Other', 'Other')])

    # Perpetrator History of Violence
    history_of_violence = forms.ChoiceField(label=_('Has the perpetrator been violent before?'), choices=[('Yes', 'Yes'), ('No', 'No'), ('Not Sure', 'Not Sure')], required=True)
    previous_incidents = forms.CharField(label=_('If yes, please provide details of previous incidents:'), widget=forms.Textarea, required=False)
    previous_incidents_reported = forms.ChoiceField(label=_('Has the victim reported these incidents to the authorities before?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    action_taken = forms.CharField(label=_('If yes, what action was taken?'), widget=forms.Textarea, required=False)

    # Risk Assessment
    access_to_weapons = forms.ChoiceField(label=_('Does the perpetrator have access to weapons?'), choices=[('Yes', 'Yes'), ('No', 'No'), ('Not Sure', 'Not Sure')], required=True)
    threats_made = forms.ChoiceField(label=_('Has the perpetrator made threats to harm the victim or others?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    victim_afraid = forms.ChoiceField(label=_('Is the victim afraid of the perpetrator?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    children_dependents = forms.ChoiceField(label=_('Does the victim have children or dependents?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    safe_place = forms.ChoiceField(label=_('Is the victim currently in a safe place?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)

    # Support and Resources
    medical_attention = forms.ChoiceField(label=_('Does the victim need immediate medical attention?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    safe_housing = forms.ChoiceField(label=_('Does the victim need assistance with safe housing?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    legal_help = forms.ChoiceField(label=_('Does the victim require legal assistance?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    counseling = forms.ChoiceField(label=_('Does the victim require counseling?'), choices=[('Yes', 'Yes'), ('No', 'No')], required=True)
    other_support = forms.CharField(label=_('Any other support needed:'), widget=forms.Textarea, required=False)

    # Additional Information
    additional_information = forms.CharField(label=_('Any other information the victim would like to provide:'), widget=forms.Textarea, required=False)

    # Consent
    consent = forms.BooleanField(label=_('I consent to the information provided being shared with the relevant authorities and organizations for the purpose of seeking assistance and support.'), required=True)

    def clean(self):
        cleaned_data = super().clean()
        cleaned_data['risk_ranking'] = rank_user_risk(cleaned_data)
        return cleaned_data

def rank_user_risk(cleaned_data):
    """
    Function to rank the risk level based on the user's Spain Domestic Violence Form Submission.
    The function checks various sections of the form to determine the risk level using the same risk
    classifications as the Police of Spain with VioGen.
    """
    risk_level = 'unappreciated'

    yes_count = 0
    risk_assessment_fields = [
        'access_to_weapons', 
        'threats_made', 
        'victim_afraid', 
        'children_dependents', 
        'safe_place'
    ]
    for field in risk_assessment_fields:
        if cleaned_data.get(field) == 'Yes':
            yes_count += 1

    history_of_violence = cleaned_data.get('history_of_violence') == 'Yes'
    support_needed = (
        cleaned_data.get('medical_attention') == 'Yes' or 
        cleaned_data.get('safe_housing') == 'Yes' or 
        cleaned_data.get('legal_help') == 'Yes' or 
        cleaned_data.get('counseling') == 'Yes'
    )
    other_responses_length = sum([
        len(cleaned_data.get('previous_incidents', '').split()), 
        len(cleaned_data.get('action_taken', '').split()), 
        len(cleaned_data.get('other_support', '').split()), 
        len(cleaned_data.get('additional_information', '').split())
    ])

    if yes_count > 2:
        risk_level = 'extreme'
    elif yes_count == 2:
        risk_level = 'high'
    elif history_of_violence or support_needed:
        risk_level = 'medium'
    elif other_responses_length > 10:
        risk_level = 'medium'
    elif yes_count == 1:
        risk_level = 'low'
    
    return risk_level

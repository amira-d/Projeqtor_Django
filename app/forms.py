from django.forms import ModelForm, TextInput
from .models import Project

class PredictForm(ModelForm):
    class Meta:
        model = Project
        fields = ['Title','Client','Sector','ProjectDuration','NbPhases']
        widgets = {
            'Title': TextInput(attrs={'class' : 'input', 'placeholder' : 'Title'}),
            'Client': TextInput(attrs={'class' : 'input', 'placeholder' : 'Client'}),
            'Sector': TextInput(attrs={'class' : 'input', 'placeholder' : 'Sector'}),
            'Project Duration': TextInput(attrs={'class' : 'input', 'placeholder' : 'Project Duration'}),
            'Number of Phases': TextInput(attrs={'class' : 'input', 'placeholder' : 'Number of Phases'}),

            
        } #updates the input class to have the correct Bulma class and placeholder
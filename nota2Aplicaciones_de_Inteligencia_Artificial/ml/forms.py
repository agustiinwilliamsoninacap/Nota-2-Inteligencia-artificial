from django import forms


class InsuranceForm(forms.Form):
    edad = forms.IntegerField(label="Edad (años)", min_value=18, max_value=100, initial=35)
    sexo = forms.ChoiceField(
        label="Sexo",
        choices=[("male", "Hombre"), ("female", "Mujer")],
        initial="female"
    )
    imc = forms.FloatField(label="Índice de masa corporal (IMC, kg/m²)", min_value=10, max_value=60, initial=27.5)
    hijos = forms.IntegerField(label="Número de hijos", min_value=0, max_value=10, initial=0)
    fumador = forms.ChoiceField(label="¿Fuma actualmente?", choices=[("yes", "Sí"), ("no", "No")], initial="no")
    region = forms.ChoiceField(
        label="Región de residencia",
        choices=[
            ("southwest", "Suroeste"),
            ("southeast", "Sureste"),
            ("northwest", "Noroeste"),
            ("northeast", "Noreste"),
        ],
        initial="southeast"
    )


class DiabetesForm(forms.Form):
    embarazos = forms.IntegerField(label="Número de embarazos", min_value=0, max_value=20, initial=1)
    glucosa = forms.FloatField(label="Nivel de glucosa en sangre (mg/dL)", min_value=0, initial=120)
    presion = forms.FloatField(label="Presión arterial diastólica (mmHg)", min_value=0, initial=70)
    pliegue = forms.FloatField(label="Espesor del pliegue cutáneo (mm)", min_value=0, initial=20)
    insulina = forms.FloatField(label="Nivel de insulina (µU/mL)", min_value=0, initial=80)
    imc = forms.FloatField(label="Índice de masa corporal (kg/m²)", min_value=0, initial=28.0)
    parentesco = forms.FloatField(label="Predisposición familiar a diabetes (0–2)", min_value=0, initial=0.4)
    edad = forms.IntegerField(label="Edad (años)", min_value=1, max_value=120, initial=33)

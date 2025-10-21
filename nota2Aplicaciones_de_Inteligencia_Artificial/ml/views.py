from django.shortcuts import render
from .forms import InsuranceForm, DiabetesForm

def home(request):
    return render(request, "ml/home.html")

def insurance_view(request):
    result = None
    form = InsuranceForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data

        # Traducción de claves (español → inglés para el modelo)
        payload = {
            "age": data["edad"],
            "sex": data["sexo"],
            "bmi": data["imc"],
            "children": data["hijos"],
            "smoker": data["fumador"],
            "region": data["region"],
        }

        from .services import predict_insurance
        result = predict_insurance(payload)

    return render(request, "ml/insurance.html", {"form": form, "result": result})


def diabetes_view(request):
    result = None
    form = DiabetesForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data

        # Traducción de claves del formulario (español → inglés)
        payload = {
            "Pregnancies": data["embarazos"],
            "Glucose": data["glucosa"],
            "BloodPressure": data["presion"],
            "SkinThickness": data["pliegue"],
            "Insulin": data["insulina"],
            "BMI": data["imc"],
            "DiabetesPedigreeFunction": data["parentesco"],
            "Age": data["edad"],
        }

        from .services import predict_diabetes
        result = predict_diabetes(payload)

    return render(request, "ml/diabetes.html", {"form": form, "result": result})


def rf_report_view(request):
    import pandas as pd, pathlib
    BASE = pathlib.Path(__file__).resolve().parent.parent
    models_dir = BASE / "models"

    ins_tbl, dia_tbl = None, None
    try:
        ins_tbl = pd.read_csv(models_dir / "insurance_importances.csv").head(10).values.tolist()
    except FileNotFoundError:
        ins_tbl = []

    try:
        dia_tbl = pd.read_csv(models_dir / "diabetes_importances.csv").head(10).values.tolist()
    except FileNotFoundError:
        dia_tbl = []

    return render(request, "ml/rf_report.html", {
        "insurance": ins_tbl,
        "diabetes": dia_tbl
    })


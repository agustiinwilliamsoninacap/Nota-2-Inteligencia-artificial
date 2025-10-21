from django.core.management.base import BaseCommand
from ml.training import train_insurance, train_diabetes, load_csvs_from_models

class Command(BaseCommand):
    help = "Entrena ambos modelos y guarda artefactos en /models"

    def handle(self, *args, **kwargs):
        ins, dia = load_csvs_from_models()
        train_insurance(ins)
        train_diabetes(dia)
        self.stdout.write(self.style.SUCCESS("OK: modelos guardados en /models"))

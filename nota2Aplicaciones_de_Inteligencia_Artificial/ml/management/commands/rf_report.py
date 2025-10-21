# ml/management/commands/rf_report.py
from django.core.management.base import BaseCommand
from ml.rf_analysis import run_both_reports

class Command(BaseCommand):
    help = "Genera CSVs de importancia de características con RandomForest para ambos datasets"

    def handle(self, *args, **kwargs):
        run_both_reports()
        self.stdout.write(self.style.SUCCESS("OK: se generaron insurance_importances.csv y diabetes_importances.csv en /models"))

"""
Classification: tell the trials apart (e.g. left, right, rest) with rLDA on the
components the selected unmixing models recover.
"""
from src.classification.report import classification_report
from src.classification.run import classify

__all__ = ["classify", "classification_report"]

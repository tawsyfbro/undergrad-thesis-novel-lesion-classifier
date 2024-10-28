# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import numpy as np


class VisualizationHelper:
    @staticmethod
    def plot_training_history(history: Dict):

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: list):

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

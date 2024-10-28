# main.py
import os
from src.data.data_loader import DataLoader
from src.models.feature_extractor import FeatureExtractor
from src.models.classifier import ClassifierModel
from src.training.trainer import ModelTrainer
from src.utils.visualization import VisualizationHelper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load configurations
    model_config = 'configs/model_config.yaml'
    data_config = 'configs/data_config.yaml'

    # Initialize components
    data_loader = DataLoader(data_config)
    feature_extractor = FeatureExtractor(model_config)
    classifier = ClassifierModel(model_config)
    trainer = ModelTrainer(model_config)

    # Load and preprocess data
    logger.info("Loading dataset...")
    images, labels = data_loader.load_dataset()

    # Extract features
    logger.info("Extracting features...")
    feature_model = feature_extractor.build_model()
    features = feature_model.predict(images)

    # Prepare data for training
    X_train, X_test, y_train, y_test = trainer.prepare_data(features, labels)

    # Train and evaluate model
    if classifier.classifier_type == '1d_cnn':
        model = classifier.build_1d_cnn()
        results = trainer.train_cnn(model, X_train, y_train, X_test, y_test)
    else:
        model = classifier.build_random_forest()
        results = trainer.train_random_forest(
            model, X_train, y_train, X_test, y_test)

    # Visualize results
    if 'history' in results:
        VisualizationHelper.plot_training_history(results['history'])

    logger.info("Classification Report:")
    print(results['metrics']['classification_report'])


if __name__ == "__main__":
    main()

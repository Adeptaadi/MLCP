from src.train_clf import train_classification
from src.train_reg import train_regression

if __name__ == "__main__":

    print("\n===== TRAINING CLASSIFICATION =====")
    train_classification()

    print("\n===== TRAINING REGRESSION =====")
    train_regression()

    print("\nALL MODELS TRAINED SUCCESSFULLY!")
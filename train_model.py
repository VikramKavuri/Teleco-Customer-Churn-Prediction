from modeling import METADATA_PATH, MODEL_ONNX_PATH, train_and_persist_model


if __name__ == "__main__":
    result = train_and_persist_model()
    best_model_name = result.metadata["best_model_name"]
    deployed_model_name = result.metadata["deployed_model_name"]
    best_metrics = result.metrics[deployed_model_name]
    print(f"Saved model artifact to {MODEL_ONNX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Best evaluated model: {best_model_name}")
    print(f"Deployed model: {deployed_model_name}")
    print(
        "Validation metrics: "
        f"AUC={best_metrics['roc_auc']}, "
        f"F1={best_metrics['f1']}, "
        f"Accuracy={best_metrics['accuracy']}"
    )

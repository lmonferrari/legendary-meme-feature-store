import os
import hashlib
import time
from feature_store import create_feature_store
from model import Model


if __name__ == "__main__":
    execution_id = hashlib.md5(f"{time.time()}-{os.urandom(16)}".encode()).hexdigest()
    basedir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = "artifacts"

    os.makedirs(os.path.join(basedir, artifacts_path, execution_id), exist_ok=True)

    features = create_feature_store()
    features.to_csv(
        os.path.join(basedir, "feature_store", "feature_store.csv"),
        sep=";",
        index=False,
    )

    model = Model(X=features.drop("target", axis=1), y=features["target"])
    model.train()
    accuracy, cls_report = model.evaluate()
    model_path = os.path.join(basedir, artifacts_path, execution_id, "model.joblib")
    model.save_model(model_path)
    predict_path = os.path.join(
        basedir, artifacts_path, execution_id, "predictions.csv"
    )
    model.save_predicts(predict_path)

    run_info_path = os.path.join(
        basedir, artifacts_path, execution_id, "run_info.json"
    )
    model.save_info(
        run_info_path=run_info_path,
        run_info={
            "run_id": execution_id,
            "run_path": run_info_path,
            "accuracy": accuracy,
            "model_path": model_path,
            "predict_path": predict_path,
        },
    )

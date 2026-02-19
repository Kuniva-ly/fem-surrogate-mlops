import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path("data/processed/baseline_model.joblib")


def _required_features(model) -> list[str]:
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("Unable to infer required features from model.")


def _load_case(case_json: str | None, case_file: Path | None) -> dict:
    if case_json is None and case_file is None:
        raise ValueError("Provide --case-json or --case-file.")
    if case_json is not None and case_file is not None:
        raise ValueError("Use only one of --case-json or --case-file.")

    try:
        if case_json is not None:
            payload = json.loads(case_json)
        else:
            payload = json.loads(case_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Input case must be a JSON object.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict surrogate outputs for one new case.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to .joblib/.pkl model")
    parser.add_argument("--case-json", type=str, default=None, help="Inline JSON object for one case")
    parser.add_argument("--case-file", type=Path, default=None, help="Path to JSON file for one case")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = joblib.load(args.model_path)
    case = _load_case(args.case_json, args.case_file)

    required = _required_features(model)
    missing = [c for c in required if c not in case]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    x = pd.DataFrame([{k: case[k] for k in required}])
    pred = model.predict(x)

    output = {
        "model_path": str(args.model_path),
        "required_features": required,
        "input": {k: case[k] for k in required},
        "prediction": {
            "max_displacement_m": float(pred[0][0]),
            "max_von_mises_pa": float(pred[0][1]),
        },
    }
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

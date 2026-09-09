from ultralytics import YOLO

from config import CLASS_NAMES


def test_model_class_names_match_config():
    model = YOLO('models/modol.pt')
    assert set(model.names.keys()) == set(CLASS_NAMES.keys())
    assert set(CLASS_NAMES.values()) == set(model.names.values())

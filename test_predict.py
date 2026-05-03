from backend.app.services.ultrasound_service import ultrasound_service
import os

def test_first(directory):
    for f in os.listdir(directory):
        if f.endswith('.jpg'):
            path = os.path.join(directory, f)
            with open(path, "rb") as file:
                res = ultrasound_service.predict(file.read())
                print(f"{path}: {res}")
            break

test_first("data/test/Normal")
test_first("data/test/PCOS")

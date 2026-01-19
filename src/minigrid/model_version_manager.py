import os

class ModelVersionManager:
    def __init__(self, base_dir="models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def level_dir(self, level_id):
        path = os.path.join(self.base_dir, f"level_{level_id}")
        os.makedirs(path, exist_ok=True)
        return path

    def next_run_id(self, level_id):
        level_path = self.level_dir(level_id)
        runs = [
            d for d in os.listdir(level_path)
            if d.startswith("run_")
        ]
        return f"run_{len(runs) + 1:03d}"

    def model_path(self, level_id, run_id):
        return os.path.join(
            self.level_dir(level_id),
            f"{run_id}.zip"
        )
    
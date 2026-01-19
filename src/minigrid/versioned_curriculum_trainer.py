class VersionedCurriculumTrainer:
    def __init__(
        self,
        env_fn,
        model_fn,
        version_manager,
        env_class,
        device="cpu",
    ):
        self.env_fn = env_fn
        self.model_fn = model_fn
        self.version_manager = version_manager
        self.env_class = env_class
        self.device = device

        self.model = None
        self.current_level = None
        self.current_run_id = None

    def available_levels(self):
        return self.env_class.get_available_levels()

    def start_new_run(self, level_id=None):
        if level_id is None:
            level_id = self.available_levels()[0]

        env = self.env_fn(level_id)
        self.model = self.model_fn(env)
        self.current_level = level_id
        self.current_run_id = self.version_manager.next_run_id(level_id)

    def train_all_levels(self, timesteps_per_level=10_000):
        levels = self.available_levels()

        for i, level in enumerate(levels):
            if i == 0:
                self.start_new_run(level)
            elif level != self.current_level:
                self._switch_to_level(level)

            self.model.learn(total_timesteps=timesteps_per_level)

            self.model.save(
                self.version_manager.model_path(level, self.current_run_id)
            )

    def _switch_to_level(self, level_id):
        self.model.env.close()
        self.model.set_env(self.env_fn(level_id))
        self.current_level = level_id
        self.current_run_id = self.version_manager.next_run_id(level_id)



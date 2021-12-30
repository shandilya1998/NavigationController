from simulations.agent_model import AgentModel

class Quadruped(AgentModel):
    
    VELOCITY_LIMITS: float = 4.0

    def __init__(self, file_path: Optional[str] = 'point.xml') -> None:
        file_path = os.path.join(
            os.getcwd(),
            'assets',
            'xml',
            file_path
        )
        super().__init__(file_path, 1)

    def _set_action_space(self):
        low = np.array([0.0, -np.pi], dtype = np.float32)
        high = np.array([self.VELOCITY_LIMITS * 1.41, np.pi], dtype = np.float32)
        self.action_dim = 2 
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_obs(self):
        rgb, depth = self.sim.render(
            width = 100,
            height = 75,
            camera_name = 'mtdcam',
            depth = True
        )
        depth = 255 * (depth - 0.965) / 0.035
        depth = depth.astype(np.uint8)
        #cv2.imshow('depth', np.flipud(depth / 255.0))
        img = np.flipud(np.concatenate([
            rgb, np.expand_dims(depth, -1)
        ], -1))
        return img

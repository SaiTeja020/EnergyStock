import requests
import numpy as np

class OpenEnvClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def reset(self, seed=None, task="hard"):
        response = requests.post(
            f"{self.base_url}/reset",
            json={"seed": seed, "task": task}
        )
        response.raise_for_status()
        data = response.json()
        return self._obs_to_array(data)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action_val = [float(x) for x in action.tolist()]
        elif isinstance(action, list):
            action_val = [float(x) for x in action]
        else:
            action_val = [float(action)] * 3
            
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action_val}
        )
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        
        obs_array = self._obs_to_array(data["observation"])
        reward = data["reward"]
        terminated = data["terminated"]
        truncated = data["truncated"]
        info = data["info"]
        
        return obs_array, reward, terminated, truncated, info

    def _obs_to_array(self, obs_dict):
        return np.array([
            obs_dict["hour_of_day"],
            obs_dict["soc"],
            obs_dict["price_lmp"],
            obs_dict["p_avg"],
            obs_dict["freq_regd"],
            obs_dict["load_mw"]
        ], dtype=np.float32)

    def observation_space_shape(self):
        return (6,)

    def action_space_sample(self):
        return np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)

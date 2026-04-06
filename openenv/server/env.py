import numpy as np
from bess_rl.openenv.models import BatteryConfig, ObservationModel, ActionModel, StepResult
from bess_rl.data.pjm_dataminer import load_or_generate_data

class BESSEnvironment:
    """
    OpenEnv compliant Environment for TDD-ND BESS Co-Optimization.
    """
    def __init__(self, data_path=None):
        self.config = BatteryConfig()
        self.data_path = data_path
        self.data = None
        self.current_step = 0
        self.max_steps = 0
        self.soc = self.config.initial_soc
        self.task = "hard"

    def reset(self, seed: int = None, task: str = "hard") -> ObservationModel:
        if seed is not None:
            np.random.seed(seed)
        
        self.task = task
        self.data = load_or_generate_data(num_days=30, output_path=self.data_path, seed=seed)
        self.max_steps = len(self.data) - 1
        self.current_step = 0
        self.soc = self.config.initial_soc
        return self._get_obs()

    def _get_obs(self) -> ObservationModel:
        row = self.data.iloc[self.current_step]
        p_avg = self.data['lmp'].iloc[max(0, self.current_step - 24):self.current_step + 1].mean()
        return ObservationModel(
            hour_of_day=float(row['hour_of_day']),
            soc=float(self.soc),
            price_lmp=float(row['lmp']),
            p_avg=float(p_avg),
            freq_regd=float(row['regd']),
            load_mw=float(row['load'])
        )

    def step(self, action_model: ActionModel) -> StepResult:
        action_ps, action_ea, action_fr = action_model.action
        
        row = self.data.iloc[self.current_step]
        lmp = row['lmp']
        regd_signal = row['regd']
        load_mw = row['load']

        # Action Combining (Eq 14): a_final = clip(a_PS + a_EA + a_FR)
        a_final = float(np.clip(action_ps + action_ea + action_fr, -1.0, 1.0))

        # Determine actual power commands considering SOC limitations
        dt = 1.0 # 1 hour steps
        current_energy = self.soc * self.config.capacity_mwh

        if a_final > 0: # Charge
            p_request = a_final * self.config.max_charge_mw
            max_p_charge = ((self.config.capacity_mwh - current_energy) / self.config.efficiency_charge) / dt
            p_actual = min(p_request, max_p_charge)
            new_energy = current_energy + (p_actual * dt * self.config.efficiency_charge)
            p_charge = p_actual
            p_discharge = 0.0
        else: # Discharge
            p_request = abs(a_final) * self.config.max_discharge_mw
            max_p_discharge = (current_energy * self.config.efficiency_discharge) / dt
            p_actual = min(p_request, max_p_discharge)
            new_energy = current_energy - (p_actual * dt / self.config.efficiency_discharge)
            p_charge = 0.0
            p_discharge = p_actual

        self.soc = new_energy / self.config.capacity_mwh
        self.soc = max(0.0, min(1.0, self.soc))

        # Reward Components (Equations 7-13 from SRS)
        
        # 1. Degradation Cost cb (Eq 7-8)
        cb = self.config.cell_price / (2 * self.config.cycles * 0.8) # delta_soc assumed 0.8 DoD cycle testing equivalent
        cost_deg = p_discharge * cb * dt

        # 2. Energy Arbitrage (EA) (Eq 9)
        # Using 24 hr rolling average LMPs up to this step
        p_avg = self.data['lmp'].iloc[max(0, self.current_step - 24):self.current_step + 1].mean()
        r_ea = (lmp - p_avg) * (p_discharge - p_charge) * dt

        # 3. Frequency Regulation (FR) (Eq 2)
        net_injected = p_discharge - p_charge
        a_r = self.config.max_discharge_mw
        # Correctly compare normalized signal to normalized injection
        sc_t = max(0.0, 1.0 - abs(regd_signal - (net_injected / a_r)))
        
        r_fr = 0.0
        if sc_t >= 0.75:  # Require 75% accuracy - dense enough to learn from, strict enough to prevent idle exploit
            B_market = 300.0  # Scaled up: perfect FR should contribute ~$70k/episode to be learnable above EA noise
            r_fr = sc_t * B_market * (self.config.capacity_mwh / 150.0)

        # 4. Peak Shaving (PS)
        net_load = load_mw + p_charge - p_discharge
        peak_threshold = 20.0
        r_ps = -max(0.0, net_load - peak_threshold) * 5.0 # Penalty factor

        # Combine based on task
        if self.task == "easy":
            reward = r_ea
        elif self.task == "medium":
            reward = r_ea + r_fr
        else:
            reward = r_ea + r_fr + r_ps

        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps)
        
        info = {
            "r_ea": r_ea, "r_fr": r_fr, "r_ps": r_ps,
            "soc": self.soc, 
            "action_final": a_final,
            "action_ps": float(action_ps),
            "action_ea": float(action_ea),
            "action_fr": float(action_fr),
            "lmp": float(lmp),
            "baseline_load": float(load_mw),
            "net_load": float(net_load)
        }

        return StepResult(
            observation=self._get_obs(),
            reward=float(reward),
            terminated=terminated,
            truncated=False,
            info=info
        )

from models.dynamics.dyn_black_box import DynamicBlackBox
from models.dynamics.neural_ode import NeuralODE


class DynamicFactory:

    @staticmethod
    def create_dynamic(num_states: int,
                       num_signals: int,
                       residual: dict,
                       config: dict):
        if config["module"] in ("rnn", "transformer"):
            return DynamicBlackBox(num_states,
                                   num_signals,
                                   residual,
                                   config)
        elif config["module"] == "neural_ode":
            return NeuralODE(num_states,
                             num_signals,
                             residual,
                             config)
        else:
            raise ValueError("Invalid module type")

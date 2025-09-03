from pytorch_toolbox.loop_callback_base import LoopCallbackBase
import torch.nn.functional as F
from pytorch_toolbox.train_state import TrainingState
from torch.autograd import Variable


class DeepTrackCallback(LoopCallbackBase):
    def __init__(self, file_output_path, is_dof_only=True, update_rate=10):
        super().__init__()
        self.file_output_path = file_output_path
        self.labels = ["tx", "ty", "tz", "rx", "ry", "rz"]

    def batch(self, state: TrainingState):
        prediction = state.last_prediction[0].data.clone()
        dof_loss = F.mse_loss(Variable(prediction), Variable(state.last_target[0])).item()
        self.batch_logger["DOF"] = dof_loss
        for i, label in enumerate(self.labels):
            component_loss = F.mse_loss(Variable(prediction[:, i]), Variable(state.last_target[0][:, i])).item()
            self.batch_logger[label] = component_loss

    def epoch(self, state: TrainingState):

        self.print_batch_data(state, order=["DOF", "tx", "ty", "tz", "rx", "ry", "rz"])
        self.save_epoch_data(self.file_output_path, state)

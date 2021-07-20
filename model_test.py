import os
from time import sleep
import torch
from torch.utils.data import DataLoader, random_split
import dataset

from model import OneDimConv
from minilogue import CONTROL_TYPES, control_choices, default_control_dict
from connect import send_control_states, init_interfaces, play_note, ACTIVE_CONTROL_CODES

SAVE_DIR = './data/control_data'
MODEL_SAVE_DIR = './data/models'
MODEL_SAVE_NAME = 'conv_model_1.pth'

# To match the training / test splits with the ones used in training
torch.manual_seed(4)

# Returns the closest valid matching midi control value in [0, 127]
# from model predictions in [0,1]
def prediction_to_control(control_num, predicted_value):
  predicted_value = round(predicted_value * 127)
  control_type = CONTROL_TYPES.get(control_num)
  control_range = control_choices(control_type)
  if len(control_range) == 128:
    return predicted_value

  else:
    return min(control_range, key=lambda a: abs(predicted_value - a))


def load_model(state_path):
    model_state = torch.load(state_path)
    conv_model = OneDimConv()
    conv_model.load_state_dict(model_state)
    return conv_model


def sound_test():
    model = load_model(os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME))
    model.eval()
    _, outport = init_interfaces()
    # Create train and test splits from the custom dataset
    _, test_dataset = random_split(
        dataset.SynthSoundsDataset(SAVE_DIR), [29000, 1000]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    num_samples = test_dataset.__len__()
    control_defaults = default_control_dict()
    print(control_defaults)
    send_control_states(outport, control_defaults)
    for i, (controls, sample) in enumerate(test_loader):
        print(i)
        # Run the forward pass
        outputs = model(sample)
        ground_controls = list(controls[0].numpy())
        ground_control_dict = control_dict(ground_controls)
        ground_control_dict = {c_n: prediction_to_control(c_n, c_v) for c_n, c_v in ground_control_dict.items()}
        pred_control_dict = control_dict(list(outputs[0].detach().numpy()))
        pred_control_dict = {c_n: prediction_to_control(c_n, c_v) for c_n, c_v in pred_control_dict.items()}
        print(ground_control_dict , pred_control_dict)
        send_control_states(outport, ground_control_dict)
        sleep(.2)
        play_note(outport, 60, True)
        sleep(2)
        play_note(outport, 60, False)
        send_control_states(outport, pred_control_dict)
        sleep(1)
        play_note(outport, 60, True)
        sleep(2)
        play_note(outport, 60, False)
        sleep(2)
        

def control_dict(control_array):
    return dict(zip(ACTIVE_CONTROL_CODES, control_array))


sound_test()
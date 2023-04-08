import torch

from load_data import convert_to_feature_update, format_skills
from model import ScoreModel


# loat model
model = ScoreModel()
model.load_state_dict(torch.load('model_cpu.pth'))

def run_model(list_subject_history, subject_required):
    X = convert_to_feature_update(list_subject_history, subject_required)
    mask = torch.zeros(1, 1, 11)
    subject_required_format_ = format_skills(subject_required)
    for i in range(10):
        if subject_required_format_[i] != 0:
            mask[:,:,i] = 1
    mask[:,:,10] = 1
    output = model(input)
    return output*mask
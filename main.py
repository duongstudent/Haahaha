from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import Depends, FastAPI

import torch

from pydantic import BaseModel
from load_data import convert_to_feature_update, format_skills,format_skills_update
from model import ScoreModel

class ScoreOut(BaseModel):
    output: List[float]

class Skill(BaseModel):
    skill: Dict[str, int]

class ListSubjectHistory(BaseModel):
    list_subject_history: List[Dict[str, Union[str, int, float, Skill]]]

class SubjectRequired(BaseModel):
    subject_required: Dict[str, int]

class SubjectRequiredAndListSubjectHistory(BaseModel):
    list_subject_history: List[Dict[Union[Dict,str, int, float, Skill], Union[Dict,str, int, float, Skill]]]
    subject_required: Dict[str, int]


# load model
model = ScoreModel()
model.load_state_dict(torch.load('model_cpu.pth'))

app = FastAPI()


@app.post("/result/")
async def run_model(data: SubjectRequiredAndListSubjectHistory):
    list_subject_history = data.list_subject_history
    subject_required = data.subject_required


    input = convert_to_feature_update(list_subject_history, subject_required)
    mask = torch.zeros(1, 1, 11)
    subject_required_format_ = format_skills_update(subject_required)
    for i in range(10):
        if subject_required_format_[i] != 0:
            mask[:,:,i] = 1
    mask[:,:,10] = 1
    input = [float(i) for i in input]
    input = torch.tensor([[input]])
    output = model(input)
    output = output*mask
    output = output[0][0].tolist()
    return output


if __name__ == "__main__":
    uvicorn.run(app,port=8000,host="0.0.0.0",debug=True)

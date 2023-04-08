import json
import numpy as np

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

data = load_data('student_dict.json')
subject_university = load_data(('Subject.json'))


def take_skills(subject_history):
    skills = {}
    for subject in subject_history:
        score = subject['score']
        skill_list = subject['skills']
        for skill in skill_list:
            new_score = np.round(float(score)/10 * skill_list[skill], 2)
            if skill not in skills.keys():
                skills[skill] = new_score
            else:
                if skills[skill] < new_score:
                    skills[skill] = new_score
    return skills
    
def format_skills(subject):
    form = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 11 level
    for skill in subject['skills']:
        index = subject['skills'][skill] - 1
        form[index] += 1
    form[-1] = subject['score']
    return form

def format_skills_update(subject):
    new_subject = {}
    new_subject['skills'] = subject.copy()
    form = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 11 level
    for skill in new_subject['skills']:
        index = new_subject['skills'][skill] - 1
        form[index] += 1
    return form

def take_subject_infor(subject_name,subject_university):
    for subject in subject_university:
        if subject['name'] == subject_name:
            return subject
    return None
    

def defind_level_hard(x): # tam giac can 45 do
    x = int(x)
    acreage = 0.5 * x * x
    return acreage

def format_level_hardd(x):
    level = [0.5, 2.0, 4.5, 8.0, 12.5, 18.0, 24.5, 32.0, 40.5, 50.0]
    for i in range(len(level)):
        if x < level[i]:
            return i + 1

    
def convert_to_feature(list_subject_history, subject_required, old_skills):
    X = []
    for subject in list_subject_history:
        X += format_skills(subject)

    subject_required_copy = subject_required.copy()
    for skill in subject_required_copy['skills']:
        if skill in old_skills.keys():
            subject_required_copy['skills'][skill] = format_level_hardd(defind_level_hard(subject_required_copy['skills'][skill]) - defind_level_hard(old_skills[skill]))
    
    X += format_skills(subject_required_copy)
 
    return X

def convert_to_feature_update(list_subject_history, subject_required):
    old_skills = take_skills(list_subject_history)
    X = []
    for subject in list_subject_history:
        X += format_skills(subject)

    subject_required_copy = {}
    subject_required_copy['skills'] = subject_required.copy()
    subject_required_copy['score'] = 0
    for skill in subject_required_copy['skills']:
        if skill in old_skills.keys():
            subject_required_copy['skills'][skill] = format_level_hardd(defind_level_hard(subject_required_copy['skills'][skill]) - defind_level_hard(old_skills[skill]))
    
    X += format_skills(subject_required_copy)
 
    return X


def take_student_data(student, subject_university):
    list_subject = student['subject_history']
    data = []
    for subject in list_subject:
        new_list_subject = [i for i in list_subject if i != subject]
        old_skills = take_skills(new_list_subject)
        subject_required = take_subject_infor(subject['name'],subject_university)
        X = convert_to_feature(new_list_subject, subject_required, old_skills)
        y = format_skills(subject) # lenght = 11
        data.append((X, y))
    return data

def load_data_student():
    new_dataset = []

    for student in data:
        list_data = take_student_data(student,subject_university)
        new_dataset += list_data
    
    return new_dataset
    

if __name__ == '__main__':
    new_dataset = []

    for student in data:
        list_data = take_student_data(student,subject_university)
        new_dataset += list_data

    print(len(new_dataset))



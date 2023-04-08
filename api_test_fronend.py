import requests
import streamlit as st
import json



headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

st.title("Test API")

if st.button("Start Process"):
    list_subject_history = [
    {
        "_id": "20",
        "index": 19,
        "name": "Software Engineering",
        "skills": {
            "Math": 7,
            "C": 8,
            "Logical Thinking": 8,
            "Creativity": 7,
            "Teamwork": 8,
            "Self_Taught": 8
        },
        "score": 6.0
    }
    ]
    subject_required = {'Math': 8,
                    'Java': 9,
                    'Logical Thinking': 8,
                    'Creativity': 7,
                    'Teamwork': 8,
                    'Self_Taught': 8}

    # res = requests.post(f"http://0.0.0.0:8000/predict/", data=json.dumps({"list_subject_history": list_subject_history, "subject_required": subject_required}), headers=headers)
    res = requests.post(
    f"http://0.0.0.0:8000/result/",
    data=json.dumps({"duong": "test_value"}),
    headers=headers
)

    res_dict = res.json()      
    st.json(res_dict)

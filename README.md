## Student Exam Performance Predictor

This project is a Flask-based web application that predicts a student’s Math score based on key performance indicators like gender, ethnicity, parental education level, lunch type, test preparation, and reading/writing scores.

 Features
*  Predicts math scores using trained ML model
*  User-friendly web interface built with HTML + CSS
*  Real-time prediction with Flask backend
*  Handles form inputs and returns accurate results
*  Input validations and field selection dropdowns

  ## Project Structure

  STUDENT_PERFORMANCE_END_TO_END/
│
├── artifacts/ # Stores raw and processed CSV files
│ ├── train.csv
│ └── test.csv
│
├── logs/ # Logs from training/inference
├── notebook/ # Jupyter Notebooks (exploration/training)
│
├── src/
│ ├── components/ # Data ingestion, model training etc.
│ ├── pipeline/ # Prediction pipeline logic
│ ├── exception.py # Custom exceptions
│ ├── logger.py # Logging functions
│ └── utils.py # Utility functions
│
├── templates/
│ ├── index.html # Landing page
│ └── homee.html # Prediction form page (consider renaming)
│
├── app.py # Flask app entrypoint
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── setup.py # (Optional) For pip install
├── .gitignore # Hides unnecessary files from Git
└── venv/ # Python virtual environment


## Model Details

The model is trained using a regression algorithm to predict math score based on:

Gender

Race/Ethnicity

Parental Level of Education

Lunch Type

Test Preparation Course

Reading Score

Writing Score

## Tech Stack
Frontend: HTML, CSS (with animations)

Backend: Python, Flask

Modeling: Scikit-learn, Pandas, NumPy

Deployment Ready: Configurable for any cloud

## How to Run Locally

1) Clone the repository
   git clone https://github.com/your-username/student-performance-predictor.git
   cd student-performance-predictor

2) Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate

3) Install dependencies
   pip install -r requirements.txt

4) Run the Flask app
   python app.py

5) Open in browser
   http://127.0.0.1:5000/


## Screenshots

Home Page

![image](https://github.com/user-attachments/assets/097c1987-7cf4-4d79-a8be-9c1884537cfb)


Prediction

![image](https://github.com/user-attachments/assets/1f43be1f-604e-483a-8853-61f9eb51e4f3)





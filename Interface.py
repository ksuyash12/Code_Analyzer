
import warnings
warnings.filterwarnings('ignore')
GOOGLE_API_KEY="AIzaSyAcTq9IBJnJ4d6KEStnnWlPPqxh6DobA-I"
code= """
## Step 1: Data Collection
import pandas as pd

# Load the dataset (replace 'user_book_interactions.csv' with the path to your dataset)
data = pd.read_csv('user_book_interactions.csv')

# Display the first few rows of the dataset
data.head()

## Step 2: Data Preprocessing
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset

# Load the data into Surprise library's format
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['user_id', 'book_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

## Step 3: Model Training
from surprise import SVD
from surprise.model_selection import cross_validate

# Train an SVD model
model = SVD()
cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

## Step 4: Model Evaluation
# Train the model on the full training set
model.fit(trainset)

# Evaluate the model on the test set
predictions = model.test(testset)

from surprise import accuracy
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

## Step 5: Deployment
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.json['user_id'])
    top_n = 10  # Number of recommendations to return

    # Get a list of all book ids
    book_ids = data['book_id'].unique()

    # Predict ratings for all books
    predictions = [model.predict(user_id, book_id) for book_id in book_ids]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top N recommended books
    recommended_books = [pred.iid for pred in predictions[:top_n]]

    return jsonify({'recommended_books': recommended_books})

if __name__ == '__main__':
    app.run(debug=True)
"""

requirements= """
the goal of this task is to create a system that can recommend books to users based on their reading history. The system should utilize collaborative filtering techniques and machine learning algorithms to accurately recommend books that a user might be interested in.

## Requirements
1. **Data Collection**:
   - Use the provided dataset of user-book interactions in a CSV file.
   - Each interaction includes the user ID, book ID, and rating given by the user to the book.

2. **Data Preprocessing**:
   - Clean and preprocess the data (e.g., handling missing values, encoding categorical features).
   - Convert the data into a suitable format for machine learning models (e.g., user-item interaction matrix).

3. **Model Training**:
   - Split the data into training and testing sets.
   - Train a collaborative filtering model (e.g., matrix factorization, neural collaborative filtering) on the training data to predict user ratings for books.

4. **Model Evaluation**:
   - Evaluate the performance of the trained model on the testing set using metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

5. **Deployment**:
   - Implement a simple web interface where users can input their user ID and get book recommendations.
   - Use a framework such as Flask or FastAPI for the web interface.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**: Pandas, Scikit-learn, Surprise, Flask/FastAPI
- **Dataset**: Provided CSV file with user-book interactions

## Additional Information
- Ensure the code is well-documented with comments and docstrings.
- Write a brief report summarizing your approach, the models used, and the results obtained.

## Example Usage
- Input: User ID: 123
- Output: Recommended Books: ["The Catcher in the Rye", "To Kill a Mockingbird", "1984"]

"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentExecutor
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.9, google_api_key=GOOGLE_API_KEY)
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)


from langchain import PromptTemplate

def Accuracy():
  evaluation_prompt = """ You are an expert in code evaluation. Evaluate the following code based on the given requirements and criteria:
Requirements: {requirements}
Code: {code}
Evaluate the code based on the Accuracy metric and suggest possible changes to be made to make model more effective (you dont have to run the code just analyze it):
Provide a short feedback on this metric and suggest improvements.
"""
  output= llm.invoke(evaluation_prompt.format(code=code, requirements=requirements))
  return output.content

def Creativity():
  evaluation_prompt2 = """You are an expert in code evaluation. Evaluate the following code based on the given requirements and criteria (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Creativity" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output =llm.invoke(evaluation_prompt2.format(code=code, requirements=requirements))
  return output.content


def Loopholes():
  evaluation_prompt3 = """You are an expert in code evaluation.You will be given one metric. Evaluate the following code based on the given requirements and metric provided (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Loophole" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output= llm.invoke(evaluation_prompt3.format(code=code, requirements=requirements))
  return output.content

def Efficiency():
  evaluation_prompt3 = """You are an expert in code evaluation.You will be given one metric. Evaluate the following code based on the given requirements and metric provided (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Efficiency" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output=llm.invoke(evaluation_prompt3.format(code=code, requirements=requirements))
  return output.content
def Scalability():
  evaluation_prompt3 = """You are an expert in code evaluation.You will be given one metric. Evaluate the following code based on the given requirements and metric provided (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Scalability" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output = llm.invoke(evaluation_prompt3.format(code=code, requirements=requirements))
  return output.content

def Extensibility():
  evaluation_prompt3 = """You are an expert in code evaluation.You will be given one metric. Evaluate the following code based on the given requirements and metric provided (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Extensibility-(Ease of adding new features or modifying existing ones.
Design patterns that support extensibility.)" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output =llm.invoke(evaluation_prompt3.format(code=code, requirements=requirements))
  return output.content
def Modularity_Reusability():
  evaluation_prompt3 = """You are an expert in code evaluation.You will be given one metric. Evaluate the following code based on the given requirements and metric provided (you dont have to run the code just analyze it):
Requirements: {requirements}
Code: {code}
Evaluate the code based on the "Modularity and Reusability-Code is organized into functions and classes,
Reusable components are identified and abstracted.)" metric and suggest possible changes to be made considering that metric only:
Provide a short feedback on this metric and suggest improvements.
"""
  output = llm.invoke(evaluation_prompt3.format(code=code, requirements=requirements))
  return output.content

import streamlit as st
gradient_background_css = """
<style>
    .main {
        background: linear-gradient(to right, #1E1E1E, #3A3A3A);
        color: white;
    }
    .css-18e3th9 {
        background: linear-gradient(to right, #1E1E1E, #3A3A3A);
    }
    .css-1aumxhk {
        background: linear-gradient(to right, #1E1E1E, #3A3A3A);
    }
</style>
"""
st.markdown(gradient_background_css, unsafe_allow_html=True)


def show_description():
    st.title("Project Description")
    st.write("""
    ## Code Evaluation Project
    This project evaluates code snippets based on various criteria to ensure quality and performance.
    
    ## Tech Stack
    - **Langchain** and **Google Gemini 1.5 Pro Model** as the LLM framework and Generative AI.
    - A Python agent using the **PythonREPL Tool** for better understanding of the python code written.

    ### Evaluation Metrics:
    - **Accuracy**: How well the code meets the requirements.
    - **Creativity**: The innovative solutions and approaches used in the code.
    - **Loopholes**: Any potential vulnerabilities or weaknesses in the code.
    - **Efficiency**: The performance and optimization of the code.
    - **Scalability**: The ability of the code to handle larger datasets or more complex scenarios.
    - **Extensibility**: How easily the code can be extended to add new features.
    - **Modularity & Reusability**: The structure and reusability of the code across different projects.

    Use the sidebar to navigate to different evaluation metrics and see the detailed analysis of the code.
    """)
def evaluate_code(metric):
    if metric == "Accuracy":
        st.subheader("Accuracy Evaluation")
        st.write(Accuracy())
    elif metric == "Creativity":
        st.subheader("Creativity Evaluation")
        st.write(Creativity())
    elif metric == "Loopholes":
        st.subheader("Loopholes Evaluation")
        st.write(Loopholes())
    elif metric == "Efficiency":
        st.subheader("Efficiency Evaluation")
        st.write(Efficiency())
    elif metric == "Scalability":
        st.subheader("Scalability Evaluation")
        st.write(Scalability())
    elif metric == "Extensibility":
        st.subheader("Extensibility Evaluation")
        st.write(Extensibility())
    elif metric == "Modularity & Reusability":
        st.subheader("Modularity & Reusability Evaluation")
        st.write(Modularity_Reusability())
page = st.sidebar.selectbox(
    "Navigate",
    ("Project Description", "Evaluate Code")
)
if page == "Project Description":
    show_description()
elif page == "Evaluate Code":

    metric = st.sidebar.selectbox(
        "Select the metric to evaluate",
        ("Accuracy", "Creativity", "Loopholes", "Efficiency", "Scalability", "Extensibility", "Modularity & Reusability")
    )
    evaluate_code(metric)
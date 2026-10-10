 
import pandas as pd 
 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, classification_report 
 
# Step 1: Load the dataset 
data = pd.read_csv("student_pass_prediction_dataset.csv") 
 
print("===== STUDENT DATASET =====") 
print(data.head()) 
print("\nTotal Students:", len(data)) 
 
# Step 2: Separate input features and target 
X = data[ 
    ["hours_studied", "attendance", "previous_marks"] 
] 
 
y = data["result"] 
 
# Step 3: Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y 
) 
 
print("\nTraining Records:", len(X_train)) 
print("Testing Records:", len(X_test)) 
 
# Step 4: Create and train the model 
model = DecisionTreeClassifier( 
    max_depth=3, 
    random_state=42 
) 
 
model.fit(X_train, y_train) 
 
print("\nModel Training Completed!") 
 
# Step 5: Evaluate the model 
y_pred = model.predict(X_test) 
 
accuracy = accuracy_score(y_test, y_pred) 
 
print("\n===== MODEL EVALUATION =====") 
print("Actual Results:", list(y_test)) 
print("Predicted Results:", list(y_pred)) 
print("Model Accuracy:", round(accuracy * 100, 2), "%") 
 
print("\nClassification Report:") 
print(classification_report( 
    y_test, 
    y_pred, 
    zero_division=0 
)) 
 
# Step 6: Enter a new student's details 
print("\n===== STUDENT PASS PREDICTION =====") 
 
try: 
    hours = float(input("Enter study hours per day: ")) 
    attendance = float(input("Enter attendance percentage: ")) 
 
    marks_input = input( 
        "Enter previous marks separated by spaces: " 
    ) 
 
    marks_list = [ 
        float(mark) for mark in marks_input.split() 
    ] 
 
    if not marks_list: 
        raise ValueError("Enter at least one mark.") 
 
    if not 0 <= hours <= 24: 
        raise ValueError("Study hours must be between 0 and 24.") 
 
    if not 0 <= attendance <= 100: 
        raise ValueError("Attendance must be between 0 and 100.") 
 
    if any(mark < 0 or mark > 100 for mark in marks_list): 
        raise ValueError("Each mark must be between 0 and 100.") 
 
    # Calculate average of previous marks 
    average_marks = sum(marks_list) / len(marks_list) 
 
    print("Average Previous Marks:", round(average_marks, 2)) 
 
    # Prepare new student data 
    new_student = pd.DataFrame([{ 
        "hours_studied": hours, 
        "attendance": attendance, 
        "previous_marks": average_marks 
    }]) 
 
    # Predict result 
    prediction = model.predict(new_student) 
 
    print("\n===== PREDICTION RESULT =====") 
    print("Study Hours:", hours) 
    print("Attendance:", attendance, "%") 
    print("Average Previous Marks:", round(average_marks, 2)) 
    print("Predicted Result:", prediction[0]) 
 
except ValueError as error: 
    print("Invalid Input:", error)      
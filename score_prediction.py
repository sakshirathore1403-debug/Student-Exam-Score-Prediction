import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error


# Basic knowledge about data
df=pd.read_csv(r"C:\Users\Deepa Rathore\OneDrive\Desktop\python libraries\scikit_learn\student_exam_scores.csv")
print(df.head(),"\n")
print("information about data")
print(df.info(),"\n")
print(df.describe(),"\n")
print(df.drop_duplicates(subset=None,keep='first',inplace=True,ignore_index=False))


# ENCODING OF DATA
le=LabelEncoder()
df['student_id']=le.fit_transform(df['student_id'])
    


# Train-Test_Split of Data

X=df[['student_id','hours_studied','sleep_hours','attendance_percent','previous_scores']]
y=df[['exam_score']]

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)


# Train data model
model=LinearRegression()
model.fit(X_train,y_train)

    
# Test model
def predi(df):
    y_pred=model.predict(df)
    return y_pred

y_pred=predi(X_test)



# evaluate the model performance

print("minimum value of Final Exam Score prediction : ",y_pred.min())
print("maximum value of Final Exam Score prediction",y_pred.max())


MAE=mean_absolute_error(y_test,y_pred)
print(f"\nmean absolute error is : {MAE}")

MSE=mean_squared_error(y_test,y_pred)
print(f"mean square error is : {MSE}")



# Ploting of actual and predicted final exam score
plt.scatter(X_test['hours_studied'],y_test,label="actual value")
plt.scatter(X_test['hours_studied'],y_pred,label="predict value")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Exam Score")
plt.title("Actual vs Predicted Final Exam Score")
plt.legend(loc="upper left")
plt.grid()
# plt.savefig("exam_score_prediction.pdf",dpi=100,bbox_inches='tight')
plt.show()


# Usser input
st.title("Student Score Prediction Model")
st.text("To predict your marks please provide some information")
with st.form("my_form"):
    data={
    'student_id' : [st.text_input('enter your enrollment no. : ')],
    'hours_studied' : [st.number_input('enter your study hours per Day : ')],
    'sleep_hours' : [st.number_input('enter Sleeping Hours : ')],
    'attendance_percent' : [st.number_input('enter attendance percentage : ')],
    'previous_scores' : [st.number_input('enter your previous exam score : ')],
    }
    submit=st.form_submit_button("Submit")

if submit:
    df1=pd.DataFrame(data)
    df1['student_id']=le.fit_transform(df1['student_id'])
    y_p=predi(df1)
    st.write("Your marks should be ",y_p)



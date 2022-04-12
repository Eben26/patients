 
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
#print("starting to load saved model")estimamodeltor
model = load_model('savedkmeans.pkl')
#print("loaded the saved model")






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    st.sidebar.info('This app is created to predict if an employee will leave the company')
    st.sidebar.success('https://www.pycaret.org')
    #st.sidebar.image(image_office)
    st.title("Predicting employee leaving")
#     if add_selectbox == 'Online':
#         satisfaction_level=st.number_input('satisfaction_level' , min_value=0.1, max_value=1.0, value=0.1)
#         last_evaluation =st.number_input('last_evaluation',min_value=0.1, max_value=1.0, value=0.1)
#         number_project = st.number_input('number_project', min_value=0, max_value=50, value=5)
#         time_spend_company = st.number_input('time_spend_company', min_value=1, max_value=10, value=3)
#         Work_accident = st.number_input('Work_accident',  min_value=0, max_value=50, value=0)
#         promotion_last_5years = st.number_input('promotion_last_5years',  min_value=0, max_value=50, value=0)
#         salary = st.selectbox('salary', ['low', 'high','medium'])
#         output=""
#         input_dict={'satisfaction_level':satisfaction_level,'last_evaluation':last_evaluation,'number_project':number_project,'time_spend_company':time_spend_company,'Work_accident': Work_accident,'promotion_last_5years':promotion_last_5years,'salary' : salary}
#         input_df = pd.DataFrame([input_dict])
#         if st.button("Predict"):
#             output = predict(model=model, input_df=input_df)
#             output = str(output)
#         st.success('The output is {}'.format(output))
    #if add_selectbox == 'Batch':
    #file_upload='awsws'
    file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload) 
        #print(data.shape)
        predictions = predict_model(model,data=data)
        #predictions.to_csv('sasasas.csv')
        #print(predictions.columns)
#         c=input ('enter the relationship name')
#         print(c)
#             for i in range(len(got_data)):
#                 temp = got_data.iloc[i,:].to_dict()
#                 src = temp["Id"]
#                 dst = str(temp['Label'])
#                 temp_1 = {j:temp[j] for j in temp if j not in ['ID', "Label"]}
#                 src_title = "<br>".join(str(temp_1).split(','))

#                 got_net.add_node(src, src, color="#00CC00", title=src_title)
#                 got_net.add_node(dst, dst, color="#FFCC00", title="Cluster_"+dst)
#                 got_net.add_edge(src, dst, color='#0000CC')

        st.write(predictions)
#             output = predict(model=model, input_df=data)
#             output = str(output)
#             print(output)
def main():
    run()

if __name__ == "__main__":
  main()

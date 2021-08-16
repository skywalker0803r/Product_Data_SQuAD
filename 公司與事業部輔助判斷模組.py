import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import set_seed,model_predict,Collection_method
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline
import torch
import plotly.express as px
from sklearn.decomposition import PCA

seed = set_seed(42)

# bert 
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load('./models/Product_Data_SQuAD_model_2144.pt'))
model.eval()
nlp = pipeline('question-answering', model=model.to('cpu'), tokenizer=tokenizer)
nlp_feature_extraction = pipeline("feature-extraction", model=model.to('cpu'), tokenizer=tokenizer)
train_df = pd.read_csv('./data/preprocess_for_SQUAD.csv')
train_df = train_df[['45A']].rename(columns={'45A':'string_X_train'})
max_text = max(train_df['string_X_train'].values,key=len)
train_text = train_df['string_X_train'].sample(100).tolist()
train_text.append(max_text)
train_emb = nlp_feature_extraction(train_text, padding="longest", truncation=True, max_length=40)
pca = PCA(n_components=2)
pca.fit(train_emb)
train_emb = pca.transform(train_emb)

# 產品集合
df = pd.read_excel('./data/台塑企業_ 產品寶典20210303.xlsx',engine='openpyxl')
產品集合 = set(df['品名'].values)
品名2部門 = dict(zip(df['品名'],df['公司事業部門']))
品名2代號 = dict(zip(df['品名'],df['公司代號']))

# UI
st.title('公司與事業部輔助判斷模組')
text_input = st.text_area('輸入信用狀資訊判斷所屬事業部')
df = pd.DataFrame(index=[0])
df.loc[0,'string_X_train'] = text_input
test_emb = nlp_feature_extraction([text_input,max_text], padding="longest", truncation=True, max_length=40)
test_emb = pca.transform(test_emb)

# rule or bert?
text_output = Collection_method(df, 產品集合)
print(text_output)
mode = 'rule'
if str(text_output.values[0][0])== "not find":
    mode = 'bert'
    text_output = model_predict(nlp,df)

# show predict result
button = st.button('predict')

if not button:
    st.stop()

if button:
    product = text_output.values[0][0]
    st.text(f'產品名稱:{product}')
    st.text(f'預測方式:{mode}')
    try:
        st.text(f'部門:{品名2部門[product]}')
        st.text(f'部門代號:{品名2代號[product]}')
    except:
        st.text(f'找不到所屬部門')
    
    df1 = pd.DataFrame()
    df1['x'] = train_emb[:,0]
    df1['y'] = train_emb[:,1]
    df1['type'] = 'trainset'
    df1['string_X_train'] = train_text
    
    df2 = pd.DataFrame()
    df2['x'] = test_emb[:,0]
    df2['y'] = test_emb[:,1]
    df2['type'] = 'testset'
    df2['string_X_train'] = [text_input,max_text]

    df3 = df1.append(df2.head(1))

    fig = px.scatter(df3, x="x", y="y", color="type", hover_data=['string_X_train'])

    st.plotly_chart(fig)




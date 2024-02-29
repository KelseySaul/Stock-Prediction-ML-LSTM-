import numpy as np 
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import nltk

model = load_model(r'C:\Users\Kelsey\OneDrive\Documents\stockk\Stock Predictions Model.keras')

st.header('Stock Price Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-30'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)
data["sentiment"] = np.random.uniform(-1,1,len(data))
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'b')
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []
 
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale 
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label = 'Original Price')
plt.plot(y, 'g', label ='Predicted Price' )
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)


from sklearn.linear_model import LinearRegression




import yahoo_fin.stock_info as si

import streamlit as st
from newsapi import NewsApiClient

def get_latest_news(symbol):
    
    # Initialize News API client
    newsapi = NewsApiClient(api_key='a8a3f0148846480293ade4cdc79d22ca')  
    keywords = ['stock', 'finance', 'market', 'investment', 'trading']
    
   
    query = symbol + ' ' + ' OR '.join(keywords)
    
   
    news = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
    
    
    filtered_news = [article for article in news['articles'] if symbol.lower() in article['title'].lower() or symbol.lower() in article['content'].lower()]
    
    return filtered_news

def main():
    st.header("Latest Stock News Headlines")
    
    symbol = st.text_input('Enter Stock Symbol')
    if symbol:
        try:
            news = get_latest_news(symbol)
            if news:
                st.write(f"### News Headlines for {symbol}")
                for article in news:
                    title = article['title']
                    url = article['url']
                    st.write(f"**{title}** [Read more]({url})")
            else:
                st.write("No news found for the specified symbol.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")

if __name__ == "__main__":
    main()


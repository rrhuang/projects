import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Conv1D, GlobalMaxPooling1D, Dropout, Attention, Reshape
from transformers import TFDistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder




def create_model(price_features,tech_features):
    tweet_input = Input(shape=(None,), dtype=tf.int32, name='tweet_input')
    bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    bert_embeddings = bert_model(tweet_input)[0]
    cnn_layer = Conv1D(64, kernel_size=3, activation='relu')(bert_embeddings)
    cnn_layer = GlobalMaxPooling1D()(cnn_layer)


    price_input = Input(shape=(len(price_features),), name='price_input')
    lstm_price = LSTM(64, return_sequences=True)(tf.expand_dims(price_input, axis=1))
    attention_price = Attention()([lstm_price, lstm_price])
    lstm_price_output = LSTM(64)(attention_price)

    tech_input = Input(shape=(len(tech_features),), name='tech_input')
    lstm_tech = LSTM(64, return_sequences=True)(tf.expand_dims(tech_input, axis=1))
    attention_tech = Attention()([lstm_tech, lstm_tech])
    lstm_tech_output = LSTM(64)(attention_tech)

    concatenated = Concatenate(axis=1)([cnn_layer, lstm_price_output, lstm_tech_output])
    dense1 = Dense(64, activation='relu')(concatenated)
    dropout = Dropout(0.5)(dense1)
    output = Dense(3, activation='softmax')(dropout)

    model = Model(inputs=[tweet_input, price_input, tech_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



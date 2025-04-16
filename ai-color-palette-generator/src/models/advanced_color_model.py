from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

class ColorPaletteGenerator:
    def __init__(self, max_words=1000, max_len=50, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None

    def create_dataset(self):
        emotions_data = [
            {'emotion': 'happy energetic', 'brief': 'vibrant and uplifting mood', 'colors': ['#FFD700', '#FFA500', '#FF4500', '#FF6B6B', '#FFB6C1']},
            {'emotion': 'calm peaceful', 'brief': 'serene and tranquil atmosphere', 'colors': ['#E0FFFF', '#B0E0E6', '#87CEEB', '#ADD8E6', '#F0F8FF']},
            {'emotion': 'passionate romantic', 'brief': 'warm and intimate feeling', 'colors': ['#FF1493', '#FF69B4', '#FFB6C1', '#FFC0CB', '#FFE4E1']},
            {'emotion': 'professional serious', 'brief': 'corporate and trustworthy look', 'colors': ['#000080', '#483D8B', '#4B0082', '#2F4F4F', '#708090']},
            {'emotion': 'natural organic', 'brief': 'earthy and sustainable theme', 'colors': ['#228B22', '#32CD32', '#90EE90', '#98FB98', '#F5DEB3']},
            {'emotion': 'luxurious elegant', 'brief': 'sophisticated and premium feel', 'colors': ['#800080', '#4B0082', '#483D8B', '#FFD700', '#C0C0C0']},
            {'emotion': 'playful creative', 'brief': 'fun and imaginative design', 'colors': ['#FF69B4', '#00CED1', '#FFD700', '#FF6347', '#98FB98']},
            {'emotion': 'mysterious dark', 'brief': 'enigmatic and deep mood', 'colors': ['#2F4F4F', '#483D8B', '#4B0082', '#800080', '#8B4513']},
            {'emotion': 'fresh clean', 'brief': 'crisp and minimal style', 'colors': ['#F0FFFF', '#E0FFFF', '#FFFFFF', '#F5FFFA', '#F0F8FF']},
            {'emotion': 'bold dramatic', 'brief': 'striking and impactful design', 'colors': ['#FF0000', '#000000', '#FFD700', '#4B0082', '#800080']}
        ]

        texts = [f"{d['emotion']} {d['brief']}" for d in emotions_data]
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)

        y = np.array([[int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255] 
                      for d in emotions_data for color in d['colors']])
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def build_model(self):
        text_input = Input(shape=(self.max_len,))
        
        embedding = Embedding(self.max_words, self.embedding_dim)(text_input)
        lstm = LSTM(128, return_sequences=True)(embedding)
        lstm = LSTM(64)(lstm)
        
        dense1 = Dense(256, activation='relu')(lstm)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        output = Dense(15, activation='sigmoid')(dropout2)
        
        self.model = Model(inputs=text_input, outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test = self.create_dataset()
        
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def generate_palette(self, emotion, brief):
        text = f"{emotion} {brief}"
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        predictions = self.model.predict(padded)[0]
        
        colors = []
        for i in range(0, len(predictions), 3):
            r = int(predictions[i] * 255)
            g = int(predictions[i+1] * 255)
            b = int(predictions[i+2] * 255)
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        
        return colors

if __name__ == '__main__':
    generator = ColorPaletteGenerator()
    history = generator.train(epochs=50)
    
    emotion = "happy energetic"
    brief = "modern and vibrant website design"
    palette = generator.generate_palette(emotion, brief)
    print(f"Generated color palette for {emotion} ({brief}):")
    print(palette)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np

class EnhancedColorModel:
    def __init__(self):
        self.max_words = 5000
        self.max_len = 50
        self.embedding_dim = 100
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.model = None
        self.color_scaler = MinMaxScaler()

    def create_dataset(self):
        emotions_data = [
            {'emotion': 'joyful energetic', 'brief': 'vibrant celebration mood', 'colors': ['#FFD700', '#FFA500', '#FF4500', '#FF6B6B', '#FFB6C1']},
            {'emotion': 'peaceful serene', 'brief': 'tranquil meditation space', 'colors': ['#E0FFFF', '#B0E0E6', '#87CEEB', '#ADD8E6', '#F0F8FF']},
            {'emotion': 'romantic loving', 'brief': 'intimate warm atmosphere', 'colors': ['#FF1493', '#FF69B4', '#FFB6C1', '#FFC0CB', '#FFE4E1']},
            {'emotion': 'confident professional', 'brief': 'corporate trustworthy image', 'colors': ['#000080', '#483D8B', '#4B0082', '#2F4F4F', '#708090']},
            {'emotion': 'innovative creative', 'brief': 'modern tech forward', 'colors': ['#00CED1', '#4682B4', '#9370DB', '#20B2AA', '#48D1CC']},
            {'emotion': 'luxurious premium', 'brief': 'high end sophisticated', 'colors': ['#800080', '#4B0082', '#483D8B', '#FFD700', '#C0C0C0']},
            {'emotion': 'natural organic', 'brief': 'eco friendly sustainable', 'colors': ['#228B22', '#32CD32', '#90EE90', '#98FB98', '#F5DEB3']},
            {'emotion': 'fresh spring', 'brief': 'new beginning growth', 'colors': ['#98FB98', '#90EE90', '#00FA9A', '#3CB371', '#2E8B57']},
            {'emotion': 'earthy grounded', 'brief': 'natural rustic feel', 'colors': ['#8B4513', '#A0522D', '#6B4423', '#8B7355', '#CD853F']},
            {'emotion': 'mysterious dramatic', 'brief': 'enigmatic intense mood', 'colors': ['#2F4F4F', '#483D8B', '#4B0082', '#800080', '#8B4513']},
            {'emotion': 'nostalgic vintage', 'brief': 'retro classic style', 'colors': ['#DEB887', '#D2B48C', '#BC8F8F', '#F4A460', '#DAA520']},
            {'emotion': 'playful whimsical', 'brief': 'fun imaginative spirit', 'colors': ['#FF69B4', '#00CED1', '#FFD700', '#FF6347', '#98FB98']},
            {'emotion': 'clean minimal', 'brief': 'simple modern aesthetic', 'colors': ['#F0FFFF', '#E0FFFF', '#FFFFFF', '#F5FFFA', '#F0F8FF']},
            {'emotion': 'bold impactful', 'brief': 'strong statement design', 'colors': ['#FF0000', '#000000', '#FFD700', '#4B0082', '#800080']},
            {'emotion': 'subtle elegant', 'brief': 'refined understated look', 'colors': ['#F5F5F5', '#E8E8E8', '#DCDCDC', '#D3D3D3', '#C0C0C0']}
        ]

        augmented_data = self._augment_color_data(emotions_data)
        
        texts = [f"{d['emotion']} {d['brief']}" for d in augmented_data]
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)

        y = self._prepare_color_data(augmented_data)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _augment_color_data(self, data):
        augmented = []
        for entry in data:
            augmented.append(entry)
            variation = {
                'emotion': entry['emotion'],
                'brief': entry['brief'],
                'colors': self._create_color_variations(entry['colors'])
            }
            augmented.append(variation)
        return augmented

    def _create_color_variations(self, colors):
        variations = []
        for color in colors:
            r = int(color[1:3], 16) / 255.0
            g = int(color[3:5], 16) / 255.0
            b = int(color[5:7], 16) / 255.0
            
            variation = [
                max(0, min(1, r + np.random.uniform(-0.1, 0.1))),
                max(0, min(1, g + np.random.uniform(-0.1, 0.1))),
                max(0, min(1, b + np.random.uniform(-0.1, 0.1)))
            ]
            
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(variation[0] * 255),
                int(variation[1] * 255),
                int(variation[2] * 255)
            )
            variations.append(hex_color)
        return variations

    def _prepare_color_data(self, data):
        color_data = []
        for entry in data:
            palette = []
            for color in entry['colors']:
                rgb = sRGBColor(
                    int(color[1:3], 16) / 255.0,
                    int(color[3:5], 16) / 255.0,
                    int(color[5:7], 16) / 255.0
                )
                lab = convert_color(rgb, LabColor)
                palette.extend([lab.lab_l, lab.lab_a, lab.lab_b])
            color_data.append(palette)
        
        color_data = np.array(color_data)
        return self.color_scaler.fit_transform(color_data)

    def build_model(self):
        text_input = Input(shape=(self.max_len,))
        embedding = Embedding(self.max_words, self.embedding_dim)(text_input)
        attention = MultiHeadAttention(num_heads=8, key_dim=self.embedding_dim)(embedding, embedding)
        attention = LayerNormalization()(attention + embedding)
        
        lstm1 = LSTM(128, return_sequences=True)(attention)
        lstm1 = LayerNormalization()(lstm1 + attention)
        
        lstm2 = LSTM(64)(lstm1)
        
        dense1 = Dense(256, activation='relu')(lstm2)
        dropout1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(128, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        output = Dense(75, activation='sigmoid')(dropout2)
        
        self.model = Model(inputs=text_input, outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test = self.create_dataset()
        
        if self.model is None:
            self.build_model()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )
        
        return history

    def generate_palette(self, emotion, brief):
        text = f"{emotion} {brief}"
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        prediction = self.model.predict(padded)
        lab_colors = self.color_scaler.inverse_transform(prediction)[0]
        
        palette = []
        for i in range(0, len(lab_colors), 3):
            lab = LabColor(lab_colors[i], lab_colors[i+1], lab_colors[i+2])
            rgb = convert_color(lab, sRGBColor)
            
            rgb_values = [
                max(0, min(1, rgb.rgb_r)),
                max(0, min(1, rgb.rgb_g)),
                max(0, min(1, rgb.rgb_b))
            ]
            
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb_values[0] * 255),
                int(rgb_values[1] * 255),
                int(rgb_values[2] * 255)
            )
            palette.append(hex_color)
        
        return palette

if __name__ == '__main__':
    model = EnhancedColorModel()
    model.build_model()
    history = model.train()
    
    emotion = "peaceful serene"
    brief = "tranquil meditation space"
    palette = model.generate_palette(emotion, brief)
    print(f"Generated color palette for {emotion} ({brief}):")
    print(palette)
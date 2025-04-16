from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np

class ColorPaletteModel:
    def __init__(self, emotions, color_palettes):
        self.emotions = emotions
        self.color_palettes = color_palettes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Embedding(input_dim=len(self.emotions), output_dim=8, input_length=1),
            LSTM(16),
            Dense(len(self.color_palettes[0]), activation='softmax')  # Assuming each palette has the same number of colors
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs)

    def generate_color_palette(self, emotion):
        emotion_index = np.array([self.emotions.index(emotion)])
        prediction = self.model.predict(emotion_index)
        return self.color_palettes[np.argmax(prediction)]

# Example usage
if __name__ == '__main__':
    emotions = ['happy', 'sad', 'angry', 'calm']
    color_palettes = [
        ['#FFFF00', '#FFD700', '#FFA500'],  # Happy
        ['#0000FF', '#4682B4', '#708090'],  # Sad
        ['#FF0000', '#8B0000', '#B22222'],  # Angry
        ['#00FFFF', '#7FFFD4', '#E0FFFF']   # Calm
    ]

    model = ColorPaletteModel(emotions, color_palettes)
    # Dummy training data
    X_train = np.array([[0], [1], [2], [3]])  # Example indices for emotions
    y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])  # One-hot encoded palettes

    model.train(X_train, y_train, epochs=10)

    # Generate a color palette
    emotion = 'happy'
    palette = model.generate_color_palette(emotion)
    print(f"Color palette for {emotion}: {palette}")
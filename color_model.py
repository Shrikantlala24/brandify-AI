import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np

# Example dataset: emotions and corresponding color palettes
emotions = ['happy', 'sad', 'angry', 'calm']
color_palettes = [
    ['#FFFF00', '#FFD700', '#FFA500'],  # Happy
    ['#0000FF', '#4682B4', '#708090'],  # Sad
    ['#FF0000', '#8B0000', '#B22222'],  # Angry
    ['#00FFFF', '#7FFFD4', '#E0FFFF']   # Calm
]

# Convert emotions to numerical data
emotion_to_index = {emotion: idx for idx, emotion in enumerate(emotions)}
emotion_indices = np.array([emotion_to_index[emotion] for emotion in emotions])

# Define a simple model
model = Sequential([
    Embedding(input_dim=len(emotions), output_dim=8, input_length=1),
    LSTM(16),
    Dense(3, activation='softmax')  # Assuming 3 colors per palette
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy training data (for illustration purposes)
X_train = emotion_indices
y_train = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # One-hot encoded palettes

# Train the model
model.fit(X_train, y_train, epochs=10)

# Function to generate color palette
def generate_color_palette(emotion):
    emotion_index = np.array([emotion_to_index[emotion]])
    prediction = model.predict(emotion_index)
    # Convert prediction to color palette (simplified)
    return color_palettes[np.argmax(prediction)]

# Example usage
emotion = 'happy'
palette = generate_color_palette(emotion)
print(f"Color palette for {emotion}: {palette}")
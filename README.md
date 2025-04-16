# Color Palette Generator

This is an AI-powered color palette generator that creates harmonious color combinations based on emotions and descriptions.

## Setup and Usage

1. First, install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Flask application:
```bash
python app.py
```

3. **Important**: Before generating palettes, you need to train the model first!
   - Send a POST request to `/train` endpoint
   - Wait for the training to complete (this may take a few minutes)
   - After successful training, you can start generating palettes

4. To generate a palette:
   - Send a POST request to `/generate` endpoint with JSON body:
   ```json
   {
       "emotion": "joyful energetic",
       "brief": "vibrant celebration mood"
   }
   ```

## Example Emotions and Briefs

Here are some example combinations you can try after training:

- Emotion: "peaceful serene", Brief: "tranquil meditation space"
- Emotion: "confident professional", Brief: "corporate trustworthy image"
- Emotion: "natural organic", Brief: "eco friendly sustainable"
- Emotion: "mysterious dramatic", Brief: "enigmatic intense mood"
- Emotion: "clean minimal", Brief: "simple modern aesthetic"

## Note

The model must be trained before generating palettes. If you try to generate without training first, you'll get an error. The training process creates a sophisticated color psychology dataset and trains a neural network to understand the relationships between emotions and color combinations.
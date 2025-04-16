# AI Color Palette Generator

This project is an AI-powered color palette generator that creates harmonious color combinations based on user-defined emotions and design briefs. It utilizes color psychology principles and machine learning techniques to generate aesthetically pleasing color palettes.

## Features

- **Emotion-Based Color Generation**: Users can input specific emotions and design briefs to receive tailored color palettes.
- **Machine Learning Model**: The model is trained on a dataset that links emotions to color psychology, allowing for intelligent color selection.
- **User-Friendly Interface**: A simple web interface for users to interact with the model and generate color palettes.

## Project Structure

- `src/`: Contains the main application code.
  - `app.py`: The main Flask application file that handles user requests and serves the web interface.
  - `models/`: Contains the machine learning models used for color generation.
    - `enhanced_color_model.py`: The enhanced model that incorporates advanced techniques for color prediction.
    - `advanced_color_model.py`: An alternative model with different architecture for color generation.
    - `color_model.py`: A simpler model for basic color palette generation.
  - `templates/`: Contains HTML templates for rendering the web interface.
    - `index.html`: The main HTML file for the user interface.
  - `static/`: Contains static files such as CSS styles.
    - `styles.css`: The stylesheet for the web application.

- `data/`: Contains datasets used for training the models.
  - `dataset.csv`: The dataset linking emotions to color palettes.

- `requirements.txt`: Lists the Python dependencies required for the project.

- `README.md`: This file, providing an overview and instructions for the project.

- `.gitignore`: Specifies files and directories to be ignored by Git.

## Setup and Usage

1. **Install Dependencies**: Run the following command to install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. **Start the Application**: Launch the Flask application using:
   ```
   python src/app.py
   ```

3. **Train the Model**: Before generating palettes, you need to train the model. Send a POST request to the `/train` endpoint.

4. **Generate a Palette**: After training, you can generate a color palette by sending a POST request to the `/generate` endpoint with a JSON body containing the emotion and design brief.

## Example Usage

- **Emotion**: "joyful energetic"
- **Brief**: "vibrant celebration mood"

## Note

Ensure that the model is trained before attempting to generate palettes. The training process may take some time depending on the dataset size and model complexity.
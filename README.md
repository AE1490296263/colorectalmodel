# Colorectal Prediction Model

This is a Flask-based application for predicting colorectal cancer and polyp probabilities using biomarker data. Users can input their data to receive predictions and view SHAP explanation plots.

## Project Structure

## Features

- Input biomarker data for prediction
- Display probabilities for colorectal cancer and polyps
- Generate and display SHAP explanation plots

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`.

## Deployment

This application can be deployed on Render or other cloud services. Ensure the application listens on `0.0.0.0` and retrieves the port number from the environment variable.

## Contributing

Contributions are welcome! Please submit a Pull Request or report issues.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

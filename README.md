
# AgriCare: Crop Disease Diagnosis

AgriCare is a comprehensive solution for diagnosing crop diseases using AI, providing a user-friendly iOS application to capture and analyze leaf images. The system comprises three main components: an iOS app, an API, and a deep learning AI model.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

AgriCare: Crop Disease Diagnosis is your go-to app for maintaining healthy crops. Whether you're a professional farmer or a home gardener, our app helps you protect your valuable crops from diseases with ease and efficiency.

## Features

- **AI-Powered Disease Detection**: Simply take a photo of the affected part of your crop, and our advanced AI algorithms will analyze the image to identify the disease.
- **Comprehensive Crop Support**: Support for a wide range of crops including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Squash, Strawberry, and Tomato.
- **Instant Diagnosis**: Get quick and accurate results to make timely decisions and prevent the spread of diseases.
- **Expert Remedies**: Receive tailored remedies and treatment options for each identified disease to ensure effective management and recovery of your crops.
- **User-Friendly Interface**: Designed for ease of use, AgriCare makes disease diagnosis straightforward and accessible for everyone.

## Architecture

The AgriCare system is composed of three main parts:

1. **iOS App (SwiftUI)**: Developed with SwiftUI, the iOS app provides users with an easy-to-use interface for scanning and diagnosing plant diseases from leaf images.
   - Directory: `iOS/AgriCare`

2. **API (Flask)**: The API acts as the intermediary between the iOS app and the deep learning model. It handles image uploads and returns disease diagnosis results.
   - Directory: `API`

3. **Deep Learning AI Model**: The AI model is based on the InceptionV3 architecture, achieving 98.5% accuracy in diagnosing crop diseases from leaf images.
   - Directory: `AI Model`

## Setup

### Prerequisites

- Xcode (for iOS app development)
- Python 3.x (for API and AI model)
- AWS account (for deploying the AI model on EC2)
- Virtual environment (recommended)

### iOS App

1. Open the `AgriCare.xcodeproj` in Xcode.
2. Build and run the project on your iOS device or simulator.

### API

1. Navigate to the `API` directory.
2. Create a virtual environment and activate it:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Flask server:
   ```sh
   python app.py
   ```

### AI Model

1. Navigate to the `AI Model` directory.
2. Run the `PlantDiseaseClassifierFinal.ipynb` in a Jupyter Notebook.

### Deployment on AWS EC2

1. Launch an EC2 instance and set up your environment.
2. Clone the repository and navigate to the API directory.
3. Set up the virtual environment and install dependencies as described above.
4. Deploy the Flask API to handle incoming requests and serve predictions from the AI model.

## Usage

1. Open the AgriCare app on your iOS device.
2. Capture a clear photo of the affected crop leaf.
3. Submit the photo for analysis.
4. Receive an instant diagnosis with detailed information about the disease and expert recommendations for treatment.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

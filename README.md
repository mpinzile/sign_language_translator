# Sign Language Translation Solution

[ASL Alphabet Reference] (https://d.newsweek.com/en/full/1394686/asl-getty-images.jpg)

## Overview

The Sign Language Translation Solution is a computer vision-based project that translates sign language gestures into text. Currently, it can translate 9 letters (A to I) from sign language to text. The project utilizes various libraries and tools, including OpenCV, CVZONE, TensorFlow, MediaPipe, NumPy, and Tkinter. The machine learning model used for translation was trained using Teachable Machine.

## Project Components

The repository contains several main files, each serving a specific purpose:

1. **Data Collection:** This component is responsible for collecting data used to train the translation model. It captures sign language gestures for the letters A to I.

2. **Gesture Detection and Translation:** This component detects and displays the translated letter corresponding to the sign language gesture in real-time using the trained model.

3. **Sentence Generation:** The "Generate Sentence" component allows users to compose sentences by selecting letters through sign language gestures. It includes a graphical user interface (GUI) window where sentences can be constructed using gestures for the letters A to I.

## How to Use the "Generate Sentence" Component

To use the "Generate Sentence" feature of this project:

1. Clone the repository:
   ```bash
   git clone https://github.com/mpinzile/sign_language_translator.git
   ```
2. Run the "generate_sentence.py" file, which opens a Tkinter window with the following functionalities:

   - A textarea to display the composed sentence.

3. Follow the on-screen instructions to compose sentences using sign language gestures for letters A to I.

## Future Improvements

The Sign Language Translation Solution is an ongoing project with several planned enhancements:

- **Expanded Alphabet**: We aim to include translations for all letters from A to Z and expand the available gestures to include numbers (0 to 9) and special control signs for functions like "OK," "Delete," and "Space."

- **Text-Based Interfaces**: Integration with text-based interfaces to enable communication with a wider audience.

- **Model Enhancement**: Continuously improving the accuracy and efficiency of the machine learning model for better translation results.

## Contributors

1. David Mpinzile
2. William Frank
3. Elisha Stanley

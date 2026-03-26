# Week 3 - Language Model from PDF

## Overview
In this task, I built a simple language model using text extracted from a PDF book.

## Steps
- Extracted text from PDF using PyPDF2
- Converted text into tokens using tiktoken
- Created training data with context length = 100
- Split data into training and validation sets (80/20)
- Built and trained a neural network using TensorFlow/Keras
- Generated new text using the trained model

## Result
The model can generate text based on learned patterns, although output is not perfect due to simple architecture.

## Tools Used
- Python
- PyPDF2
- tiktoken
- TensorFlow / Keras
- Google Colab

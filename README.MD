# DSPY DEMO (IU TECHNICAL EXCHANGE)

This project implements a Chain of Thought (CoT) approach to solve math problems from the GSM8K dataset using DSPy and OpenAI's GPT-3.5-turbo model.

## Features

- Utilizes DSPy framework for natural language processing tasks
- Implements Chain of Thought reasoning
- Uses OpenAI's GPT-3.5-turbo model
- Includes evaluation and optimization components

## Setup

1. Install required packages:
   ```
   pip install dspy openai
   ```

2. Set up OpenAI API key in your environment variables.

## Usage

Run `main.py` to:
1. Initialize the CoT model
2. Evaluate the baseline model
3. Optimize the model using BootstrapFewShotWithRandomSearch
4. Save the compiled model

## Key Components

- `CoT`: Implements the Chain of Thought module
- `Evaluate`: Assesses model performance on the dev set
- `BootstrapFewShotWithRandomSearch`: Optimizes the model using few-shot learning

## Output

The optimized model is saved as `iu_demo.json`.

## Note

Ensure you have necessary permissions and credits to use the OpenAI API.
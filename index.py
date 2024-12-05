from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_KEY"))

print("API Key Loaded:", os.getenv("GEMINI_KEY"))

training_data = [
    {"text_input": "1", "output": "2"},
    {"text_input": "3", "output": "4"},
    {"text_input": "-3", "output": "-2"},
    {"text_input": "twenty two", "output": "twenty three"},
    {"text_input": "two hundred", "output": "two hundred one"},
    {"text_input": "ninety nine", "output": "one hundred"},
    {"text_input": "8", "output": "9"},
    {"text_input": "-98", "output": "-97"},
    {"text_input": "1,000", "output": "1,001"},
    {"text_input": "10,100,000", "output": "10,100,001"},
    {"text_input": "thirteen", "output": "fourteen"},
    {"text_input": "eighty", "output": "eighty one"},
    {"text_input": "one", "output": "two"},
    {"text_input": "three", "output": "four"},
    {"text_input": "seven", "output": "eight"},
  ]

# try:
#     with open("dataset/mydata.jsonl", "r") as file:
#         for line in file:
#             line = line.strip()  # Strip any unnecessary whitespace
#             if line:  # Skip empty lines
#                 try:
#                     training_data.append(json.loads(line))
#                 except json.JSONDecodeError as e:
#                     print(f"Error decoding line: {e}")
# except FileNotFoundError:
#     print("File not found")

print(training_data)

base_model = "models/gemini-1.5-flash-001-tuning"

operation = genai.create_tuned_model(
    display_name="increment_model",  
    source_model=base_model,
    epoch_count=20,  
    batch_size=4,   
    learning_rate=0.001,  
    training_data=training_data, 
)

for status in operation.wait_bar():
    time.sleep(10) 

result = operation.result()
print(f"Fine-tuning completed. Model name: {result.name}")

model = genai.GenerativeModel(model_name=result.name)
response = model.generate_content("III")  
print("Model Output:", response.text)

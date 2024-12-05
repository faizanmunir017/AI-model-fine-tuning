from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

client = OpenAI()


try:
    file_obj = client.files.create(
        file=open("dataset/mydata.jsonl", "rb"),
        purpose="fine-tune"
    )
    file_id = file_obj.id  
    print("File Id is:", file_id)
except Exception as e:
    print(f"Error uploading file: {e}")
    exit()


try:
    fine_tune_job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-3.5-turbo" 
    )
    print("Fine-tune job started.")
    print(fine_tune_job)

except Exception as e:
    print(f"Error starting fine-tune job: {e}")
    exit()


model_id=""

while True:
    try:
        fine_tune_job = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
        if fine_tune_job.status == "succeeded":
            model_id = fine_tune_job.fine_tuned_model
            print(f"Fine-tuned Job: {fine_tune_job.fine_tuned_model}")
            break
        elif fine_tune_job.status == "failed":
            print("Fine-tuning job failed")
            print(fine_tune_job)
            break
        else:
            print(f"Job still processing... Status: {fine_tune_job.status}")
            time.sleep(5)
    except Exception as e:  
        print()
        print(f"Error retrieving fine-tune job status: {e}")
        break


try:
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
            {"role": "user", "content": "What's the capital of France?"}
        ]
    )
    print(completion.choices[0].message)
except Exception as e:
    print(f"Error generating completion: {e}")

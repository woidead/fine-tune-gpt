from openai import OpenAI
client = OpenAI()

file_id = openai.File.create(
  file=open("training_examples.jsonl", "rb"),
  purpose='fine-tune'
).id
job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")

job_id = job.id

model_name_pre_object = openai.FineTuningJob.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)

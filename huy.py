# # from openai import OpenAI
# # import openai
# # client = OpenAI()
# # openai.api_base="https://aiproxy.9qw.ru"
# # openai.api_key = """sk-dev-azis-ask3l5qgkldebflk"""
# # file_id =client.files.create(
# #   file=open("training_examples.jsonl", "rb"),
# #   purpose="fine-tune"
# # ).id

# # job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo-1106")

# # job_id = job.id


# # print(job_id)

# # import openai

# # # Set the API key and API base

# # # Create the file
# # response = openai.File.create(
# #   file=open("training_examples.jsonl", "rb"),
# #   purpose="fine-tune"
# # )
# # file_id = response.id

# # # Create the fine-tuning job
# # job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo-1106")
# # job_id = job.id

# # print(job_id)

# from openai import OpenAI
# import openai
# client = OpenAI(

# )


# file_path = r"C:\Users\программисклик2008\Desktop\codes\vadim\gpt\training_examples.jsonl"
# with open(file_path, 'rb') as file:
#     response = openai.File.create(file=file, purpose='fine-tune')
#     file_id = response.id

# # Step 4: Create a fine-tuning job
# fine_tune_response=client.fine_tuning.jobs.create(
#     training_file=file_id,
#     model="gpt-3.5-turbo-1106"  # Replace with the model you want to fine-tune
# )

# fine_tune_job_id = fine_tune_response.id

# status_response = openai.FineTune.retrieve(id=fine_tune_job_id)
# print(status_response.status)  # This will show the current status of the job
import openai
import time
import logging



openai.api_base = "https://aiproxy.9qw.ru"
openai.api_key = "sk-dev-azis-ask3l5qgkldebflk"
def configure_logging():
    """
    Configures logging settings.
    """
    logging.basicConfig(filename='output.log', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')
    return logging.getLogger()


def upload_file(file_name):
    """
    Uploads a file to OpenAI for fine-tuning.

    :param file_name: Path to the file to be uploaded.
    :return: Uploaded file object.
    """
    # Note: For a 400KB train_file, it takes about 1 minute to upload.
    file_upload = openai.File.create(file=open(file_name, "rb"), purpose="fine-tune")
    logger.info(f"Uploaded file with id: {file_upload.id}")

    while True:
        logger.info("Waiting for file to process...")
        file_handle = openai.File.retrieve(id=file_upload.id)

        if len(file_handle) and file_handle.status == "processed":
            logger.info("File processed")
            break
        time.sleep(60)

    return file_upload


if __name__ == '__main__':
    # Configure logger
    logger = configure_logging()

    file_name = r"C:\Users\программисклик2008\Desktop\codes\vadim\gpt\training_examples.jsonl"
    uploaded_file = upload_file(file_name)

    logger.info(uploaded_file)
    job = openai.FineTuningJob.create(training_file=uploaded_file.id, model="gpt-3.5-turbo-1106")
    logger.info(f"Job created with id: {job.id}")

    # Note: If you forget the job id, you can use the following code to list all the models fine-tuned.
    # result = openai.FineTuningJob.list(limit=10)
    # print(result)


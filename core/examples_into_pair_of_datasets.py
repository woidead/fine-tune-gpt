import json
import pandas as pd
from generate_dataset import prev_examples, system_message
# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples.')

# Initialize list to store training examples
training_examples = []

# Create training examples in the format required for GPT-3.5 fine-tuning
for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }
    training_examples.append(training_example)

# Save training examples to a .jsonl file
with open('training_examples.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')
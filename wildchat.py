import re
import pandas as pd
from datasets import load_dataset

user_token = "" 

# Load the dataset with no parallelism to avoid semaphore leaks
wildchat_dataset = load_dataset("allenai/WildChat-1M", token=user_token, download_mode="force_redownload")
wildchat_dataset_train = wildchat_dataset["train"]

# Ensure no parallelism when applying functions to the dataset
wildchat_dataset_train = wildchat_dataset_train.map(lambda x: x, num_proc=1)

# User and conversation metadata fields
user_metadata_fields = ['location', 'age', 'gender']
conversation_metadata_fields = ['language', 'redacted', 'toxic', 'rate', 'title', 'custom_instruction', 'status', 'redacted']

# Prepare examples from the dataset
examples = []
for i in range(wildchat_dataset_train.shape[0]):
    data = wildchat_dataset_train[i]
    conv = data["conversation"]
    user_metadata = {"location": f"{data['state']},{data['country']}"}
    conversation_metadata = {"language": data["language"], "redacted": str(data["redacted"]), "toxic": str(data["toxic"])}
    
    for field in conversation_metadata_fields:
        if field not in conversation_metadata:
            conversation_metadata[field] = ""
    
    for field in user_metadata_fields:
        if field not in user_metadata:
            user_metadata[field] = ""
    
    example = {
        "conversation_id": data["conversation_hash"], 
        "conversation": conv,
        "source": "https://huggingface.co/datasets/allenai/WildChat-1M", 
        "model_name": data["model"],
        "user_id": data["hashed_ip"], 
        "user_metadata": user_metadata,
        "timestamp": data["timestamp"], 
        "conversation_metadata": conversation_metadata
    }
    examples.append(example)

# Convert to a DataFrame
df = pd.DataFrame(examples)

# Filter conversations based on patterns
filtered_conversations = []
message_prior = ""

thank_you_pattern = re.compile(r'(?<!")\b(thank you|thanks|thank|thx|ty|sounds (good|great|awesome|amazing|excellent|perfect)|that\'s (good|great|awesome|amazing|excellent|perfect))\b(?!")', re.IGNORECASE)
sorry_pattern = re.compile(r'(?<!")\b(sorry|i apologize|apologies?|my bad)\b(?!\s+(you|you\'re))\b(?!\s+to\s+hear)', re.IGNORECASE)
response_pattern = re.compile(r'(?<!")\b(i am happy to help|my pleasure|you\'re welcome|no problem|glad to help|anytime|happy to assist)\b', re.IGNORECASE)

for index, row in df.iterrows():
    modelname = row['model_name']
    conversation = row['conversation']
    message_prior = ""
    
    for i, message in enumerate(conversation):
        message_content = message['content']

        
        # Skip messages containing words related to certain topics like emails, story, etc.
        if any(word in message_content.lower() for word in ["mail", "story", ": “", "story", "dialogue", "dear", "name", "name_1", "friend 1", "ly)"]) and any(word not in message_content.lower() for word in ["python", "class", "def", "import"]):
            message_prior = message['content']
            continue

        if message['role'] == 'user':
            # Check for "thank you" pattern
            if thank_you_pattern.search(message['content']):
                next_message = conversation[i + 1] if i + 1 < len(conversation) else None
                
                if next_message:
                    filtered_conversations.append({
                        'id': row['conversation_id'],
                       'match': next_message['content'], 
                        'current': message['content'],
                        'prior': message_prior,           
                        'model': modelname,
                        'pattern': 'thank_you',
                        'role': 'user'
                    })
                    break    
        
        if message['role'] == 'assistant':
            # Check for "sorry" pattern
            if sorry_pattern.search(message['content']):
                next_message = conversation[i + 1] if i + 1 < len(conversation) else None
                
                if next_message:
                    filtered_conversations.append({
                        'id': row['conversation_id'],
                        'match': next_message['content'], 
                        'current': message['content'],
                        'prior': message_prior,           
                        'model': modelname,
                        'pattern': 'sorry',
                        'role': 'assistant'
                    })
                    break  

            # Check for response pattern
            elif response_pattern.search(message['content']):
                next_message = conversation[i + 1] if i + 1 < len(conversation) else None
                
                if next_message:
                    filtered_conversations.append({
                        'id': row['conversation_id'],
                        'match': next_message['content'], 
                        'current': message['content'],
                        'prior': message_prior,           
                        'model': modelname,
                        'pattern': 'response',
                        'role': 'assistant'
                    })
                    break  
        message_prior = message['content']

# Convert filtered conversations to DataFrame and save to CSV
filtered_df = pd.DataFrame(filtered_conversations)
print(len(filtered_df))
filtered_df.to_csv('wildchat_v3.csv', index=False)
 

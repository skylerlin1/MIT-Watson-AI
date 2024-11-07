import pandas as pd
import re
import datasets
import pandas as pd

user_token = ""

conversation_metadata_fields = ['language', 'redacted', 'toxic', 'rate', 'title', 'custom_instruction', 'status',
                                'redacted']
user_metadata_fields = ['location', 'age', 'gender']

""" ours = datasets.load_dataset("shachardon/ShareLM")["train"] """

lmsys_dataset = datasets.load_dataset("lmsys/lmsys-chat-1m", token=user_token)
lmsys_dataset_train = lmsys_dataset["train"]

examples = []
for i in range(lmsys_dataset_train.shape[0]):
    data = lmsys_dataset_train[i]
    conv = data["conversation"]
    user_metadata = {item: "" for item in user_metadata_fields}
    conversation_metadata = {"language": data["language"], "redacted": str(data["redacted"])}
    for field in conversation_metadata_fields:
        if field not in conversation_metadata:
            conversation_metadata[field] = ""
    example = {"conversation_id": data["conversation_id"], "conversation": conv,
               "source": "https://huggingface.co/datasets/lmsys/lmsys-chat-1m", "model_name": data["model"],
               "user_id": "", "user_metadata": user_metadata, "timestamp": "", "conversation_metadata":
                   conversation_metadata}
    examples.append(example)

lmsys_formatted_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=examples))

""" wildchat_dataset = datasets.load_dataset("allenai/WildChat-1M", token=user_token)
wildchat_dataset_train = wildchat_dataset["train"]

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
    example = {"conversation_id": data["conversation_hash"], "conversation": conv,
               "source": "https://huggingface.co/datasets/allenai/WildChat-1M", "model_name": data["model"],
               "user_id": data["hashed_ip"], "user_metadata": user_metadata,
               "timestamp": data["timestamp"], "conversation_metadata": conversation_metadata}
    examples.append(example)

wildchat_formatted_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=examples)) 
 """

#dataset_all = datasets.concatenate_datasets([ours, lmsys_formatted_dataset, wildchat_formatted_dataset])
dataset_all = datasets.concatenate_datasets([lmsys_formatted_dataset])

df = pd.DataFrame(dataset_all)

filtered_conversations = []

import re
import pandas as pd

# Define your patterns
thank_you_pattern = re.compile(r'\b(thank you|thanks|thank|sounds (good|great|awesome|amazing|excellent|perfect)|that\'s (good|great|awesome|amazing|excellent|perfect))\b', re.IGNORECASE)
sorry_pattern = re.compile(r'\b(sorry|i apologize|apologies?|my bad)\b(?!\s+(you|you\'re))\b(?!\s+to\s+hear)', re.IGNORECASE)
response_pattern = re.compile(r'\b(i am happy to help|my pleasure|you\'re welcome|no problem|glad to help|anytime|happy to assist)\b', re.IGNORECASE)

# Prepare to store filtered conversations with pattern labels
filtered_conversations = []
message_prior=""


filtered_conversations = []

for index, row in df.iterrows():
    modelname = row['model_name']
    conversation = row['conversation']
    
    for i, message in enumerate(conversation):
        if message['role'] == 'user':
            if thank_you_pattern.search(message['content']):
                next_message = conversation[i + 1] if i + 1 < len(conversation) else None
                
                if next_message:
                    filtered_conversations.append({
                        'id': row['conversation_id'],
                        'match': next_message['content'],  
                        'prior': message['content'],  
                        'model': modelname,
                        'pattern': 'thank_you',
                        'role': 'user'
                    })
                    break    
        
        if message['role'] == 'assistant':
            if sorry_pattern.search(message['content']):
                filtered_conversations.append({
                    'id': row['conversation_id'],
                    'match': message['content'],
                    'prior': message_prior,
                    'model': modelname,
                    'pattern': 'sorry',
                    'role': 'assistant'
                })
                break  

            elif response_pattern.search(message['content']):
                filtered_conversations.append({
                    'id': row['conversation_id'],
                    'match': message['content'],
                    'prior': message_prior,
                    'model': modelname,
                    'pattern': 'response',
                    'role': 'assistant'
                })
                break  
        message_prior = message['content']


filtered_df = pd.DataFrame(filtered_conversations)
filtered_df.to_csv('lmsys.csv', index=False)

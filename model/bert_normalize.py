from env.importing import *
from pytorch_pretrained_bert import BertForTokenClassification, BertConfig

import toolbox.deep_learning_toolbox as dptx


input_ids = torch.LongTensor([[31, 51, 0]])
input_mask = torch.LongTensor([[1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 0]])
labels = torch.LongTensor([[0, 1, 1]])
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
num_labels = 2
model = BertForTokenClassification(config, num_labels)
logits = model(input_ids, token_type_ids, input_mask, labels=labels)
print("logits original",logits)
optimizer = dptx.get_optimizer(model.parameters(), lr=0.0001)
logits.backward()
optimizer.step()
optimizer.zero_grad()
logits = model(input_ids, token_type_ids, input_mask, labels=torch.LongTensor([[0, 1, 1]]))
print("new logits", logits)



def get_bert_name(call_name):
    if call_name == "cased":
        return "bert-cased"
    else:
        return call_name

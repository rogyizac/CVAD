from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel

def init_bert_model():
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    
    for p in bert_model.parameters():
        p.requires_grad = False
        
    return tokenizer, bert_model
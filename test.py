from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, AlbertModel

# tokenizer = AutoTokenizer.from_pretrained("./models/albert_chinese_base")
# model = AutoModelForMaskedLM.from_pretrained("./models/albert_chinese_base")


# tokenizer = BertTokenizer.from_pretrained("./models/albert_chinese_base")
# model = AlbertModel.from_pretrained("./models/albert_chinese_base")


from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_tiny", cache_dir='./models/albert_chinese_tiny')

model = AutoModelForMaskedLM.from_pretrained("voidful/albert_chinese_tiny", cache_dir='./models/albert_chinese_tiny')
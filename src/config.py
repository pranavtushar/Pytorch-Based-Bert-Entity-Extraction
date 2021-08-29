import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = r'input\entity-extraction\input\bert-base-uncased_L-12_H-768_A-12'
MODEL_PATH = "model.bin"
TRAINING_FILE = r"input\ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True        # For bert-uncased we need to lowercase
    )

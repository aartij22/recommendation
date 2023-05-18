from tokenizers import decoders
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from transformers import BertTokenizerFast
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from typing import List
from os.path import join
import json


def tokenizer(training_data: List, output_dir: str):
    """Build and save tokenizer for custom data

    Args:
        training_data (List): list of data strings
        output_dir (str): path to save tokenizer
    """
    config_path = join(output_dir, "config.json")
    vocab_path = join(output_dir, "vocab.txt")

    # tokenizer
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence(
        [NFD(), Lowercase(), StripAccents()]
    )
    bert_tokenizer.pre_tokenizer = Whitespace()

    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    # WordPieceTrainer
    trainer = WordPieceTrainer(
        vocab_size=10000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    bert_tokenizer.train_from_iterator(
        training_data, trainer, length=len(training_data)
    )
    bert_tokenizer.decoder = decoders.WordPiece(prefix="##")
    bert_tokenizer.save(config_path)

    # vocab
    print(config_path)
    bert_tokenizer = json.load(open(config_path, "r"))
    vocab = list(bert_tokenizer["model"]["vocab"].keys())

    with open(vocab_path, "w") as f:
        for item in vocab:
            f.write("%s\n" % item)

    # wrapper of tokenizer class
    new_tokenizer = BertTokenizerFast(vocab_file=vocab_path, tokenizer_file=config_path)

    new_tokenizer.save_pretrained(join(output_dir, "config/"))


if __name__ == "__main__":
    assert False, "not to be run"
    training_data = ["CROCIN TABLET 650 MG", "PAMPERS BABY DIAPERS"]
    output_dir = "./checkpoint/"

    tokenizer(training_data=training_data, output_dir=output_dir)

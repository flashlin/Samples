#from huggingface_hub import hf_hub_download
#hf_hub_download(repo_id="bert-base-cased", filename="config.json")
from transformers import BertForPreTraining, BertTokenizer, RobertaForMaskedLM, RobertaTokenizer

huggingface_hub_repo_id = "bert-base-cased"
pretrained_path = "./download_pretrained"

BertForPreTraining.from_pretrained(huggingface_hub_repo_id).save_pretrained(pretrained_path)
BertTokenizer.from_pretrained(huggingface_hub_repo_id).save_pretrained(pretrained_path)

RobertaForMaskedLM.from_pretrained(huggingface_hub_repo_id).save_pretrained(pretrained_path)
RobertaTokenizer.from_pretrained(huggingface_hub_repo_id).save_pretrained(pretrained_path)


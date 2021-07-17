# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q torch

import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import SmoothingFunction,corpus_bleu

df= pd.read_excel("/content/Articles_data_SOC.xlsx")


def bleu(ref,gen):

  """
  Args: ref- ground truth text
        gen- generated text by the transformer model
  """
  gen_bleu= gen.split()
  ref_bleu= ref.split()
  if (len(gen_bleu)<len(ref_bleu)):
    ref_bleu= ref_bleu[:len(gen_bleu)]
  else:
    gen_bleu= gen_bleu[:len(ref_bleu)]

  smooth_func= SmoothingFunction()

  score_bleu = corpus_bleu(ref_bleu, gen_bleu, smoothing_function=smooth_func.method4)

  return score_bleu


tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
# adding EOS token to pad to avoid warnings.
model= GPT2LMHeadModel.from_pretrained("gpt2",pad_token_id= tokenizer.eos_token_id)


torch.random.seed()

article_dict={}
finance_output=[]
literature_output=[]
Entertainment_output=[]
bleu_score_finance=[]
bleu_score_scliterature=[]
bleu_score_Entertainment=[]

for index,row in df.iterrows():

  finance_text= row["Finance"]
  sc_literature_text= row["Sc_Literature"]
  Entertainment_text= row["Music/Entertainment"]
  
  encoded_id_finance= tokenizer.encode(finance_text[:100],return_tensors="pt")
  encoded_id_literature= tokenizer.encode(sc_literature_text[:100],return_tensors="pt")
  encoded_id_Entertainment= tokenizer.encode(Entertainment_text[:100],return_tensors="pt")
  
  finance_output_token= model.generate(encoded_id_finance,max_length=1000,do_sample=True,top_k=30,top_p=0.95)
  scliterature_output_token= model.generate(encoded_id_literature,max_length=1000,do_sample=True,top_p=0.95,top_k=30)
  Entertainment_output_token= model.generate(encoded_id_Entertainment,max_length=1000,do_sample=True,top_p=0.95,top_k=30)

  a= tokenizer.decode(finance_output_token[0],skip_special_tokens=True)
  b= tokenizer.decode(scliterature_output_token[0],skip_special_tokens=True)
  c= tokenizer.decode(Entertainment_output_token[0],skip_special_tokens=True)
  finance_output.append(a)
  literature_output.append(b)
  Entertainment_output.append(c)
  
  bleu_score_finance.append(bleu(finance_text,a))
  bleu_score_scliterature.append(bleu(sc_literature_text,b))
  bleu_score_Entertainment.append(bleu(Entertainment_text,c))



article_dict["finance_output"]= finance_output
article_dict["Sc_literature_output"]= literature_output
article_dict["Entertainment_output"]= Entertainment_output

df= pd.DataFrame({"Finance_BLEU":bleu_score_finance,"Scientific_Literature_BLEU":bleu_score_scliterature,"Entertainment_Bleu":bleu_score_Entertainment)
df.to_csv("GPT2_BLEU_score_all_domain.csv")

#Different Search Techniques:

# #Greedy Search
# output_token= model.generate(encoded_id,max_length=50)
# greedy_output= tokenizer.decode(output_token[0],skip_special_token=True)
# print(greedy_output)

# #Beam Search
# beam_output= model.generate(encoded_id,max_length=50,num_beams=2,early_stopping=True)

# print("Beam Output:\n"+"-"*100)
# print(tokenizer.decode(beam_output[0],skip_special_tokens=True))

# #n_gram integration
# beam_outputs= model.generate(encoded_id,max_length=50,num_beams=5,early_stopping=True,no_repeat_ngram_size=2,num_return_sequences=3)

# print("Outputs:\n"+"-"*100)
# for i,beam_output in enumerate(beam_outputs):
#   print(f"Beam output{i}: {tokenizer.decode(beam_output,skip_special_tokens=True)}")
#   print("-"*50)


# # Sampling New words:
# torch.random.seed()
# beam_output= model.generate(encoded_id,max_length=50,do_sample=True,top_k=0)


# print(tokenizer.decode(beam_output[0],skip_special_tokens=True))


# # use temperature to decrease the sensitivity to low probability candidates
# torch.random.seed()
# beam_outputs= model.generate(encoded_id,max_length=50,top_k=50,top_p=0.95)


# print("Output"+"-"*100)
# print(tokenizer.decode(beam_output[0],skip_special_tokens=True))


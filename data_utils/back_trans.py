from transformers import MarianMTModel, MarianTokenizer
import tqdm
import torch
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--src_data_path", type=str, required=True)

parser.add_argument("--tar_data_path", type=str, required=True)

parser.add_argument("--src_lan", type=str, required=True)

parser.add_argument("--tar_lan",type=str, required=True)






args = parser.parse_args()

fr_model_name = '/opt/data/private/noise_master/translation_models/opus-mt-{}-{}'.format(args.src_lan,args.tar_lan)
fr_tokenizer = MarianTokenizer.from_pretrained(fr_model_name)
fr_model = MarianMTModel.from_pretrained(fr_model_name).cuda()

en_model_name = '/opt/data/private/noise_master/translation_models/opus-mt-{}-{}'.format(args.tar_lan,args.src_lan)

en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name).cuda()
device = fr_model.device

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the modelß
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    #print(template)
    src_texts = [template(text) for text in texts]
    #print(src_texts)

    
    src_input = tokenizer(src_texts, return_tensors="pt", padding=True,max_length=512,truncation=True)
    input_ids = src_input['input_ids'].to(device)
    attention_mask = src_input['attention_mask'].to(device)

    # Generate translation using model
    translated = model.generate(input_ids=input_ids,attention_mask=attention_mask)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def beam_translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    #print(template)
    src_texts = [template(text) for text in texts]
    #print(src_texts)

    src_input = tokenizer(src_texts, return_tensors="pt",  padding=True,max_length=512,truncation=True)
    input_ids = src_input['input_ids'].to(device)
    attention_mask = src_input['attention_mask'].to(device)

    # Generate translation using model
    translated = model.generate(input_ids=input_ids,attention_mask=attention_mask,num_beams=5,num_return_sequences=5)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return translated_texts

def back_translate(texts, source_lang="en", target_lang="fr"):
    # Translate from source to target language (fr)
    fr_texts = translate(texts, fr_model, fr_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language (en)
    back_translated_texts = beam_translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return fr_texts, back_translated_texts

# en_texts = ['This is so cool', 'I hated the food', 'They were very helpful']
# aug_texts = back_translate(en_texts, source_lang="en", target_lang="fr")


data = pd.read_csv(args.src_data_path, header=None)

lines = data[:][1].values.tolist()
labels = data[:][0].values.tolist()

batch_size = 16

out_sentences = []

for idx in tqdm.trange((len(lines)-1)//batch_size+1):

    batch = lines[idx*batch_size:(idx+1)*batch_size]

    aug_sentences = back_translate(batch, source_lang=args.src_lan, target_lang=args.tar_lan)
    aug_tar = aug_sentences[1]
    for i in range(len(aug_tar)//5):
        out_sentences.append(aug_tar[i*5:(i*5)+5])


out_list = []
for i in range(len(lines)):
    out_list.append([labels[i],lines[i]]+out_sentences[i])

out_list = pd.DataFrame(out_list)

out_list.to_csv(args.tar_data_path, header=False, index=False)

    # for i in range(len(labels)):

#         writer.write( aug_sentences[1][i] + "\t" + labels[i] + '\n')

# writer.close()


# print(aug_texts)


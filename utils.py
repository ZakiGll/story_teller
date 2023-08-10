from elevenlabs import generate, save, voices
from elevenlabs import set_api_key
import os
from playsound import playsound
from pydub import AudioSegment
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import random

load_dotenv(find_dotenv())
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

voices = voices()

data = pd.read_csv('name_gender.csv')
names_dataset = pd.concat([data.head(250),data.iloc[data.shape[0]-250 : -1]])

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

classifier = pipeline("ner", model=model, tokenizer=tokenizer)

detected_names=[]
for name in names_dataset['Linda']:
    detected = classifier(name)
    if detected and detected[0]['word'] == name :
        detected_names.append(name)

name_with_gender = []
for name in detected_names:
    index = data.index[data['Linda']== name].tolist()[0]
    name_with_gender.append((name,data.iloc[index].values.tolist()[1]))


female_voice_ids = []
for voice in voices:
    if 'gender' in voice.labels and voice.labels['gender'] == 'female':
        female_voice_ids.append(voice.name)
female_voice_ids.remove('Nicole')

male_voice_ids = []
for voice in voices:
    if 'gender' in voice.labels and voice.labels['gender'] == 'male':
        male_voice_ids.append(voice.name)
male_voice_ids.remove('Ethan')


def voice_selector(response):
    entities = classifier(response)
    i_per_words = [entity['word'][:] for entity in entities if entity['entity'] == 'B-PER']
    names = list(set(i_per_words))

    detected_names_with_gender = []
    for name in names:
        for name_gender in name_with_gender:
            if name == name_gender[0]:
                detected_names_with_gender.append(name_gender)
    
    names_with_voices = []
    for person in detected_names_with_gender:
        if person[1] == 'F' :
            voice = random.choice(female_voice_ids)
            names_with_voices.append((person[0],voice))
            female_voice_ids.remove(voice)
        else :
            voice = random.choice(male_voice_ids)
            names_with_voices.append((person[0],voice))
            male_voice_ids.remove(voice)
    
    return names_with_voices

def text_to_list(text):
    text = remove_empty_lines(text)
    dialogues = []
    lines = text.strip().split("\n")
    for line in lines:
        character, dialogue = line.split(": ", 1)
        dialogue = dialogue.strip().strip('"')
        dialogues.append((character, dialogue))

    return dialogues

def text_to_audio(text,names_list):
    generated_audio_chunks = []
    for character, dialogue in text:
        for name in names_list:
            if character == name[0]:
                audio = generate(
                    text = dialogue,
                    voice = name[1],
                    model = 'eleven_multilingual_v1',
                    )
                generated_audio_chunks.append(audio)
        
        
    combined_audio = b''.join(generated_audio_chunks)
    combined_audio_path = "combined_audio.mp3"
    save(combined_audio, combined_audio_path)
    print("Combined audio saved!")

def remove_empty_lines(text):
    lines = text.splitlines() 
    non_empty_lines = [line for line in lines if line.strip() != ""]  
    result = "\n".join(non_empty_lines) 
    return result


from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv, find_dotenv
import os
from utils import name_with_gender, text_to_audio, text_to_list,voice_selector
import streamlit as st

load_dotenv(find_dotenv())
chat = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-16k-0613", openai_api_key=os.getenv("OPENAI_API_KEY"))

def story_generator(text):
    messages = [
        SystemMessage(
            content=f"""You are a story teller and your role is :  1/ possibility to generate a story with multiple characters 2/ make sure to put only one point at the end of each character speech 3/ avoid to use action sentences 4/ make shure to put the name of the character when he talks 5/ the story must be a conversation 6/ Omit any concluding statements or additional text that is not part of the character conversations.  7/ in case of family members use names not father mother ... to indicate theire speech 8/ dont let an empty line between the lines 9/ the form must be name : " speech "   10/ this is the list of the allowed names: {name_with_gender} """
        ),
        HumanMessage(
            content= text +" don't let a white space  between the lines !"
        ),
    ]
    rep = chat(messages)
    text_to_audio(text_to_list(rep.content),voice_selector(rep.content))
    return rep


def main():

    st.set_page_config(page_title="The Story teller", page_icon="📖")
    st.header("Let AI bring your imaginable stories to life 📖🎧✨")
    prompt_input = st.text_input("Enter the idea of your story...")
    if prompt_input != "":
        story = story_generator(prompt_input)
        st.audio("combined_audio.mp3")
        with st.expander("The story"):
            st.write(story.content)


if __name__ == '__main__':
    main()
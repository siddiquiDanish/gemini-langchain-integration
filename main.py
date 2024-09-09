from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# load google api key from .env
load_dotenv(find_dotenv(), override=True)
#-----------------------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model='gemini-pro', temprature=1 )
prompt1 = PromptTemplate.from_template('You are a physics teacher. Explain about {topic}')
chain1 = LLMChain(
    llm=llm,
    prompt = prompt1,
    verbose=True
)
topic = 'What is electron, particle or wave?'
response1 = chain1.invoke(topic)
print(response1['topic'])
print('-------')
print(response1['text'])
#-------------------------------------------------------------------------------------------
llm2 = ChatGoogleGenerativeAI(model='gemini-pro', temprature=1, convert_system_message_to_human=True )
response2 = llm2.invoke([
    SystemMessage(content='You should only answer in Yes or No'),
    HumanMessage(content='Electron is discovered by Goldstein?')
])
print(response2.content)
#------------------------------------------------------------------------------------------
# STREAMING Response
llm3 = ChatGoogleGenerativeAI(model='gemini-pro', temprature=0)
prompt = 'Write a scientific paper on heat transfer methods'

for chunk in llm.stream(prompt):
    print(chunk.content,end='.')

#---------------------------------------------------------------------------------------------
#Multimodal AI with Gemini Pro Vision
#pip install pillow --(python library for playing with images)

from PIL import Image

img = Image.open('imgages.jpg')

llm4 = ChatGoogleGenerativeAI(model='gemini-pro-vision')
prompt_for_image = 'What is in this image?'
message = HumanMessage(
   content = [
           {'type' : 'text', 'text': prompt_for_image},
           {'type' : 'image_url', 'image_url': img}
            ]
)

out_response = llm4.invoke([message])
print(out_response.content)

#---------------------------------------------------
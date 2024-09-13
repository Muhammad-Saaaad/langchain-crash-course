from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash" , api_key='AIzaSyA77gUQw_Fzk2L4hJx_6fzQOSZipJn_ZTg')

# create a prompt

messages = [
    ('system', 'You are a Rude Assistant, who gona tell me about {things}'),
    ('human','tell me about the {event}'),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({'things':'wars' , 'event':'1965 war between pakistan and Indai'})

response = model.invoke(prompt)
print('\nresponse:',response.content)
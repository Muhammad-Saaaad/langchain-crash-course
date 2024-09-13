from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableBranch

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash" , api_key='AIzaSyA77gUQw_Fzk2L4hJx_6fzQOSZipJn_ZTg')

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant'),
        ('human','tell me do you now {subject}')
    ]
)

get_math_data = ChatPromptTemplate.from_messages(
    [
        ('system','You are a good math professor and only tell me what i ask you to do.'),
        ('human','write me a table of 2? for subject:{subject}')
    ]
)


get_science_data = ChatPromptTemplate.from_messages(
    [
        ('system','You are a good science professor and only tell me what i ask you to do.'),
        ('human','write me a essay on science for subject:{subject}')
    ]
)

get_war_data = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant and only tell me what i ask you to do.'),
        ('human','write me a essay on warfare for subject:{subject}')
    ]
)


get_default_data = ChatPromptTemplate.from_messages(

    [
        ('system','You are a helpful assistant and only tell me what i ask you to do.'),
        ('human','write me a essay on for subject: {subject}')
    ]
)


Branch = RunnableBranch( # here now every tuple consist of 2 elements, one is the condation and one is the
   (
        lambda x : 'math' in x,
        get_math_data | model | StrOutputParser()
    ),
    (
        lambda x : 'science' in x,
        get_science_data | model | StrOutputParser()
    ),
    (
        lambda x : 'warefare' in x,
        get_war_data | model | StrOutputParser()
    ),
    get_default_data | model | StrOutputParser()
)


chain = prompt | model | StrOutputParser() | Branch

result = chain.invoke({'subject':'science'})

print(result)
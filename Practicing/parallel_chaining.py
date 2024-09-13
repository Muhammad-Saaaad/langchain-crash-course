from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda , RunnableParallel


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash" , api_key='AIzaSyA77gUQw_Fzk2L4hJx_6fzQOSZipJn_ZTg')

base_prompt = ChatPromptTemplate.from_messages([
    ('system','You are a knowledgeable assistant'),
    ('human','tell me about {subject_1} and {subject_2}')
])


def get_similarity(val1 ):
    prompt= ChatPromptTemplate.from_messages(
        [
            ('system','You are a knowledgeable assistant'),
            ('human','tell me the similarity between the {val1}')
        ]
    )
    return prompt.format_prompt(val1=val1)


def get_difference(val1 ):
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system','You are a knowledgeable assistant'),
            ('human','tell me the differences between the {val1}')
        ]
    )
    return prompt.format_prompt(val1=val1)


def combine_outputs(output1 , output2):
    result = f'Similarity: \n {output1}\n\nDifferences: \n {output2}' 
    return result

similarity = (
    (RunnableLambda(lambda val1 : get_similarity(val1=val1)) | model | StrOutputParser())
)

differences = (
    (RunnableLambda(lambda val1 : get_difference(val1=val1)) | model | StrOutputParser())
)


chain = (
    base_prompt
    | model
    | StrOutputParser()
    | RunnableParallel(branches= {'similarity': similarity , 'differences' : differences})
    | RunnableLambda(lambda x : combine_outputs(x['branches']['similarity'] , x['branches']['differences']))
    )

result = chain.invoke({'subject_1':'elerticity','subject_2':'voltage'})

print(result)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# You can also get the result in this way

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnableLambda , RunnableParallel


# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash" , api_key='AIzaSyA77gUQw_Fzk2L4hJx_6fzQOSZipJn_ZTg')

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ('system','You are a helpful assistant'),
#         ('human','tell me about {subject_1} and {subject_2}')
#     ]
# )



# def get_similarities(inputdic):
#     similarity = ChatPromptTemplate.from_messages(
#         [
#             ('system','You are a helpful assistant'),
#             ('human','tell me the similarity between {subject_1} and {subject_2}')
#         ]
#     )
#     return similarity.format_prompt(**inputdic)


# def get_differences(inputdic):
#     differences = ChatPromptTemplate.from_messages(
#         [
#             ('system','You are a helpful assistant'),
#             ('human','tell me the differences between {subject_1} and {subject_2}')
#         ]
#     )
#     return differences.format_prompt(**inputdic)

# def combine(outputs):
#     return f'Similarity:\n{outputs[similarity]}\n\nDifferences:\n{outputs[differences]}'

# similarity = (RunnableLambda(get_similarities) | model | StrOutputParser())
# differences = (RunnableLambda(get_differences) | model | StrOutputParser())

# chain = (prompt | model | StrOutputParser() | RunnableParallel(similarity=similarity , differences=differences) | RunnableLambda(combine))

# result = chain.invoke({'subject_1':'Electricity' , 'subject_2':'voltage'})
# print(result)
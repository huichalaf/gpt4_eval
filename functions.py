import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def call_gpt_4_eval(model_judge, question, response):
    system_message_gpt4 = """You're an llm evaluator, your task is to evaluate in grades from 0 to 100, the score of a response to a question having in mind that the ultimate goal is to give a correct and helpfull answer to a student, the format of input will be:
        answer:
        ###
        response:
        ###
        and you should give as output, the score and only the score, only the number"""
    message = f"""
    answer:
    ###
    {question}
    response:
    ###
    {response}
    """
    response = openai.ChatCompletion.create(
        model=model_judge,
        messages=[
            {"role": "system", "content": system_message_gpt4},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']["content"]

def call_model(model, system_message, question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        temperature=0.74,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response = response['choices'][0]["message"]["content"]
    return response
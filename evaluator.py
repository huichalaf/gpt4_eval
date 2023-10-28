import openai
import os, sys
import json
import random
import tiktoken
from functions import call_gpt_4_eval, call_model
from time import time
import dotenv
dotenv.load_dotenv()
#abrimos config.json
with open(sys.argv[2]) as f:
    config = json.load(f)

start = time()
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

file_questions = sys.argv[1]

openai.api_key = os.getenv("OPENAI_API_KEY")

goal = config["goal"]
#this is a gpt-4 evaluator
system_message_gpt4 = f"""You're an llm evaluator, your task is to evaluate in grades from 0 to 100, be exigent and strict with the grades, the score of a response to a question having in mind that the ultimate goal is to {goal}, the format of input will be:
answer:
###
response:
###
and you should give as output, the score"""
system_message_chats = config["system_message"]
questions = []
with open(file_questions) as f:
    texto = f.readlines()
    for line in texto:
        questions.append(line)

print(len(questions))
#las revolvemos
random.shuffle(questions)

model_judge = config["judge_model"]
model_base = config["base_model"]
model_tested = config["tested_model"]

price_judge = config["price_judge_model"]
price_base = config["price_base_model"]
price_tested = config["price_tested_model"]

base_responses = []
test_responses = []

total_tokens_base = 0
total_tokens_test = 0
total_tokens_base_send = 0
total_tokens_test_send = 0
scores_base = []
scores_test = []
base_avg = 0
test_avg = 0
tokens_system = num_tokens_from_string(system_message_chats, "cl100k_base")
tokens_judge = num_tokens_from_string(system_message_gpt4, "cl100k_base")
amount_questions = len(questions)
tokens_system = int(tokens_system)
total_price = amount_questions * tokens_system/1000 * price_judge
total_price += amount_questions * tokens_system/1000 * price_base
average_lenght_response = 500
total_price += average_lenght_response/1000 * amount_questions * price_base
total_price += average_lenght_response/1000 * amount_questions * price_tested
total_price += tokens_judge/1000 * price_judge
total_price += amount_questions*2*average_lenght_response/1000 * price_judge
print("total price: ", round(total_price,3),"$USD")
print("Do you want to continue? (y/n)")
answer = input()
if answer == "n":
    sys.exit()

for question in questions:
    index = questions.index(question)
    print(round(index/len(questions)*100),"%")
    question = question.replace("\n", "")
    tokens_question = num_tokens_from_string(question, "cl100k_base")
    total_tokens_base_send += tokens_question
    total_tokens_test_send += tokens_question
    total_tokens_base_send += tokens_system
    total_tokens_test_send += tokens_system
    response = ''
    score = 0
    try:
        response = call_model(model_base, system_message_chats, question)
        score = call_gpt_4_eval(model_judge, question, response)
    except Exception as e:
        print(e)
        continue
    scores_base.append(float(score))
    base_responses.append(response)
    total_tokens_base += num_tokens_from_string(response, "cl100k_base")
    question += ". Let's think step by step."
    try:
        response = call_model(model_tested, "", question)
        score = call_gpt_4_eval(model_judge, question, response)
    except:
        scores_base.pop()
        base_responses.pop()
        total_tokens_base -= num_tokens_from_string(response, "cl100k_base")
        continue
    scores_test.append(float(score))
    test_responses.append(response)
    total_tokens_test += num_tokens_from_string(response, "cl100k_base")
    if index == 100:
        break


print("tokens base completion: ",total_tokens_base)
print("tokens tested completion: ",total_tokens_test)
print("tokens base send: ",total_tokens_base_send)
print("tokens tested send: ",total_tokens_test_send)
print("scores base: ", scores_base)
print("scores test: ", scores_test)
std_base = 0
std_test = 0
for score in scores_base:
    std_base += (score - sum(scores_base)/len(scores_base))**2
std_base /= len(scores_base)
std_base = std_base**(1/2)
for score in scores_test:
    std_test += (score - sum(scores_test)/len(scores_test))**2
std_test /= len(scores_test)
std_test = std_test**(1/2)
print("std base: ", std_base)
print("std test: ", std_test)
median_base = sorted(scores_base)[len(scores_base)//2]
median_test = sorted(scores_test)[len(scores_test)//2]
print("median base: ", median_base)
print("median test: ", median_test)
moda_base = max(set(scores_base), key=scores_base.count)
moda_test = max(set(scores_test), key=scores_test.count)
print("moda base: ", moda_base)
print("moda test: ", moda_test)
print("avg base: ", sum(scores_base)/len(scores_base))
print("avg test: ", sum(scores_test)/len(scores_test))

results = {}
results["base"] = {}
results["tested"] = {}
results["base"]["scores"] = scores_base
results["base"]["responses"] = base_responses
results["tested"]["scores"] = scores_test
results["tested"]["responses"] = test_responses
results["base"]["avg"] = sum(scores_base)/len(scores_base)
results["tested"]["avg"] = sum(scores_test)/len(scores_test)
results["base"]["std"] = std_base
results["tested"]["std"] = std_test
results["base"]["median"] = median_base
results["tested"]["median"] = median_test
results["base"]["moda"] = moda_base
results["tested"]["moda"] = moda_test
results["base"]["total_tokens"] = total_tokens_base
results["tested"]["total_tokens"] = total_tokens_test
results["base"]["total_tokens_send"] = total_tokens_base_send
results["tested"]["total_tokens_send"] = total_tokens_test_send
results["base"]["total_tokens_system"] = tokens_system
results["tested"]["total_tokens_system"] = tokens_system
results["base"]["total_price"] = total_price
results["tested"]["total_price"] = total_price
results["base"]["total_questions"] = amount_questions
results["tested"]["total_questions"] = amount_questions

with open(f'results.json', 'w') as f:
    json.dump(results, f)

with open(f'base_responses.txt', 'w') as f:
    for response in base_responses:
        f.write(response)
        f.write('\n')

with open(f'test_responses.txt', 'w') as f:
    for response in test_responses:
        f.write(response)
        f.write('\n')
print("total time: ", time()-start)
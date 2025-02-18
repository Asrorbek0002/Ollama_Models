from pprint import pprint
import ollama
import time

start_time=time.time()
res = ollama.chat(
    model='llava',
    messages=[
        {'role':'user',
         'content':"Identify all food and drinks in the image. Return a structured list with each item and its exact count. Use this format Item: Quantity (e.g., Apple: 3). No extra text", #'Please provide a brief summary: type of food, portion size, and approximate calories.?',
         'images':['./images1.jpg']
         }
    ]
)

end_time=time.time()
pprint(res['message']['content'])
print("Exact time:", end_time-start_time, "seconds")





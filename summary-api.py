#import os
from flask import Flask,request
#import openai
#import json

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

#import torch
  
def summarizator(name1):
    
    model_name = 'google/pegasus-xsum'
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer(name1, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    sum_name = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return sum_name


# def gptThree(text):
#     tldr_tag = "\n tl;dr:"
#     openai.api_key = "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"
#     text1=text+tldr_tag
#     response = openai.Completion.create(
#     engine="davinci",
#     #prompt=text1,
#     prompt=text1,
#     temperature=0.3,
#     max_tokens=60,
#     top_p=1.0,
#     frequency_penalty=0.0,
#     presence_penalty=0.0
#     )

#     return response

app = Flask(__name__)

@app.route("/")
def api():
    #name=request.args.get("ozet")
    #summ=summarizator(name)
      # resp_str = json.dumps(gptThree(name))
      # resp_dict=json.loads(resp_str)
      # print (resp_dict['choices'][0]['text'])
      # summ=resp_dict['choices'][0]['text']
    #return str(summ)
    return "MERHABA"
    
if __name__ =="__main__":
    #port=5000
    #app.run(host='0.0.0.0',port=port)
    app.run(debug=True)





import os
from openai import OpenAI
import retry
import json

class OpenAIClient:
    MODEL = "gpt-4"
    
    DETERMINE_SENTIMENT_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "determine_sentiment",
                "description": "Given a list of statements, determine the overall sentiment towards the subject of the statements on a scale of 1-100, 100 being most positive and 1 being most negative. Provide an explanation for why you chose the rating. Also provide a summary of the statements in 100 words or less, focusing specifically on the most important chronological events and their dates in the statements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentiment_score": {
                            "type": "number",
                            "enum": [i for i in range(1, 101)],
                            "description": "The overall sentiment rating of all the statements, 100 being most positive and 1 being most negative."
                        },
                        "sentiment_explanation": {
                            "type": "string",
                            "description": "An explanation for why you chose the sentiment score in 50 words or less."
                        },
                        "summary": {
                            "type": "string",
                            "description": "A summary of all the statements in 100 words or less, focused specifically on the most important chronological events and their dates in the statements."
                        }
                    },
                    "required": ["sentiment_score", "sentiment_explanation", "summary"]
                }
            }
        }
    ]
    
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))


    @retry.retry(tries=3, delay=2) 
    def get_sentiment_and_summary(self, query, formatted_summaries):
        
        prompt = f"I am going to give you a list of statements about {query}. Please rate the statements on a scale of 1-100, where 1 is very negative towards {query} and 100 is very positive towards {query}. These are the statements separated by newlines: {formatted_summaries}"
        print("GPT-4 Chat Complete Prompt: " + prompt)
        
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=self.DETERMINE_SENTIMENT_TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if response is None:
            print("OpenAI API call failed")
            return
        
        json_resp = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return json_resp["sentiment_score"], json_resp["sentiment_explanation"], json_resp["summary"]
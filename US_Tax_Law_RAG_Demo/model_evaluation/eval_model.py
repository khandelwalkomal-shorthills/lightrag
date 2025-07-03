from data_parser import DatasetProcessor
# from light_rag.chat_bot import ChatbotApp
from lightrag import LightRAG, QueryParam
import re
import os
from lightrag.llm import (
    gpt_4o_mini_complete,
    azure_openai_embedding,
    azure_openai_complete
)
import numpy as np
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import wrap_embedding_func_with_attrs

from openai import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)

import json



AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
# )
# async def azure_openai_llm(
#     prompt, system_prompt=None, history_messages=[], **kwargs
# ) -> str:
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": AZURE_OPENAI_API_KEY,
#     }
#     endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
#     print(endpoint)
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     if history_messages:
#         messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})

#     payload = {
#         "messages": messages,
#         "temperature": kwargs.get("temperature", 0),
#         "top_p": kwargs.get("top_p", 1),
#         "n": kwargs.get("n", 1),
#     }

#     async with aiohttp.ClientSession() as session:
#         async with session.post(endpoint, headers=headers, json=payload) as response:
#             if response.status != 200:
#                 raise ValueError(
#                     f"Request failed with status {response.status}: {await response.text()}"
#                 )
#             result = await response.json()
#             return result["choices"][0]["message"]["content"]


# @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=10),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
# )
# async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": AZURE_OPENAI_API_KEY,
#     }
#     endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_EMBEDDING_API_VERSION}"
#     print(endpoint)
#     payload = {"input": texts}

#     async with aiohttp.ClientSession() as session:
#         async with session.post(endpoint, headers=headers, json=payload) as response:
#             if response.status != 200:
#                 raise ValueError(
#                     f"Request failed with status {response.status}: {await response.text()}"
#                 )
#             result = await response.json()
#             embeddings = [item["embedding"] for item in result["data"]]
#             return np.array(embeddings)

class ChatbotApp1:
    def __init__(self):
        # Ensure the 'testing' directory exists
        self.log_dir = "../testing"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Initialize LightRAG with the correct directory
        self.rag = LightRAG(
            working_dir=self.log_dir,
            llm_model_func=azure_openai_complete,
            embedding_func=azure_openai_embedding,
        )
class ResponseEvaluator:
    def __init__(self):
        self.data_processor = DatasetProcessor()
        self.rag = ChatbotApp1().rag
        self.data_processor.process_data()

    def insert_data_in_lightrag(self):
        context_data = self.data_processor.get_context_data()
        print("Insert data in lightrag")
        self.rag.insert(context_data)

    def light_rag_query(self, prompt):
        response = self.rag.query(prompt, param=QueryParam(mode="hybrid"))
        return response
    
    
    # def compare_answer(self, golden_answer, generated_answer, query):

    #     print("correctness response \n")
    #     """
    #     Compare the generated answer with the golden (correct) answer.
    #     Evaluates the response based on factual correctness, comprehensiveness, and clarity.
    #     """
    #     prompt = f"""
    #     You are an expert evaluator tasked with comparing two answers to the same question. One answer is the correct one, and the other is the generated answer. Your job is to assess how well the generated answer (Answer 1) matches the correct answer (Answer 2) in terms of factual correctness, comprehensiveness, and clarity.

    #     Here are the two answers:

    #     **Answer 1 (Generated Answer):**
    #     {generated_answer}

    #     **Answer 2 (Correct Answer):**
    #     {golden_answer}

    #     ---

    #     Please evaluate **Answer 1** based on the following criteria:
    #     1. **Factual Correctness**: Does the generated answer contain correct facts and details in comparison to the correct answer? Rate the factual accuracy on a scale of 0 to 100, where 100 represents an answer identical to the correct answer in factual accuracy.
    #     2. **Comprehensiveness**: Does the generated answer cover all the key points and details mentioned in the correct answer? Rate the comprehensiveness on a scale of 0 to 100, where 100 represents an answer that fully covers all aspects of the correct answer.
    #     3. **Clarity**: Is the generated answer clear, well-structured, and easy to understand, as compared to the correct answer? Rate the clarity on a scale of 0 to 100, where 100 represents an answer that is equally clear as the correct answer.

    #     After evaluating these three criteria, provide an **overall correctness score** for **Answer 1**. The overall score should be calculated by averaging the scores from the three criteria (Factual Correctness, Comprehensiveness, and Clarity).

    #     Please provide the following output:
    #     - Factual Correctness: [Score from 0 to 100]
    #     - Comprehensiveness: [Score from 0 to 100]
    #     - Clarity: [Score from 0 to 100]
    #     - Overall Correctness Percentage: [Average Score from 0 to 100]

    #     ---

    #     Here is the question for reference:
    #     {query}
    #     """

    #     # Call the OpenAI API to evaluate the answers
    #     try:
    #         response = self.rag.query(prompt, param=QueryParam(mode="hybrid"))
    #         print("compare_response\n",response)
    #         if response :
    #             scores =  self.extract_score_from_section(response)
    #             return scores
    #         print("Scores..", scores)
    #         # print("compare_answer is calling \n",response)
    #         # print(type(response))
            
    #         # Check if response is a string and try to convert to JSON
    #         if isinstance(response, str):
    #             # print("compare_answer is calling \n",response)
    #             response = json.loads(response)
    #             # print("After json convert \n",response)
                

    #         # Ensure the response contains the expected structure
    #         if "choices" in response and isinstance(response["choices"], list):
    #             evaluation_text = response["choices"][0]["message"]["content"].strip()
    #             print("Generated evaluation response:", evaluation_text)

    #             # Parse the evaluation output
    #             evaluation = self.parse_evaluation(evaluation_text)
    #             print("Final Evaluation:", evaluation)
    #             return evaluation
    #         else:
    #             print(f"Unexpected response format: {response}")
    #             return None

    #     except Exception as e:
    #         print(f"Error while calling OpenAI API: {e}")
    #         return None
        
    def compare_answer(self, golden_answer, generated_answer, query):
        """
        Compare the generated answer with the golden (correct) answer.
        Evaluates the response based on factual correctness, comprehensiveness, clarity, diversity, and empowerment.
        """
        print("Correctness response \n")

        # Prepare the prompt for evaluation
        prompt = f"""
        You are an expert evaluator tasked with comparing two answers to the same question. One answer is the correct one, and the other is the generated answer. Your job is to assess how well the generated answer (Answer 1) matches the correct answer (Answer 2) in terms of factual correctness, comprehensiveness, clarity, diversity, and empowerment.

        Here are the two answers:

        **Answer 1 (Generated Answer):**
        {generated_answer}

        **Answer 2 (Correct Answer):**
        {golden_answer}

        ---

        Please evaluate **Answer 1** based on the following criteria:
        1. **Factual Correctness**: Does the generated answer contain correct facts and details in comparison to the correct answer? Rate the factual accuracy on a scale of 0 to 100, where 100 represents an answer identical to the correct answer in factual accuracy.
        2. **Comprehensiveness**: Does the generated answer cover all the key points and details mentioned in the correct answer? Rate the comprehensiveness on a scale of 0 to 100, where 100 represents an answer that fully covers all aspects of the correct answer.
        3. **Clarity**: Is the generated answer clear, well-structured, and easy to understand, as compared to the correct answer? Rate the clarity on a scale of 0 to 100, where 100 represents an answer that is equally clear as the correct answer.
        4. **Diversity**: How varied and rich is the generated answer in providing different perspectives and insights on the question? Rate the diversity on a scale of 0 to 100, where 100 represents an answer that offers a wide range of perspectives and insights.
        5. **Empowerment**: How well does the generated answer help the reader understand and make informed judgments about the topic? Rate the empowerment on a scale of 0 to 100, where 100 represents an answer that fully empowers the reader to form their own conclusions and understanding.

        After evaluating these five criteria, calculate the **overall correctness score** for **Answer 1** by averaging the scores from the five criteria. Provide the **overall correctness score** as a percentage.

        Return the following output in **valid JSON format** dont enclude any name:

        {{
            "factual_correctness": 85,
            "comprehensiveness": 90,
            "clarity": 80,
            "diversity": 75,
            "empowerment": 70,
            "overall_correctness_percentage": 80
        }}

        Here is the question for reference:
        {query}
        """

        try:
            # Call the OpenAI API to evaluate the answers
            response = self.rag.query(prompt, param=QueryParam(mode="hybrid"))
            print("Compare response:\n", response)
            
            if isinstance(response, dict):
                 return response

            if response:
                # Try to parse the response as JSON
                try:
                    response_data = json.loads(response)
                    print("Json response\n",response_data)
                except json.JSONDecodeError:
                    print("Error: Response is not in valid JSON format")
                    return None

                # Ensure the response contains the expected structure
                if isinstance(response_data, dict):
                    # Extract scores from the response
                    scores = {
                        'factual_correctness': response_data.get("factual_correctness", 0),
                        'comprehensiveness': response_data.get("comprehensiveness", 0),
                        'clarity': response_data.get("clarity", 0),
                        'diversity': response_data.get("diversity", 0),
                        'empowerment': response_data.get("empowerment", 0),
                        'overall_correctness_percentage': response_data.get("overall_correctness_percentage", 0)
                    }
                    print("Scores extracted:", scores)
                    return scores
                else:
                    print(f"Unexpected response format: {response_data}")
                    return None
            else:
                print("Error: No response received from the API")
                return None

        except Exception as e:
            print(f"Error while calling OpenAI API: {e}")
            return None

    def extract_score_from_section(self, response):
       # Define the regex patterns to extract the scores
            factual_correctness_pattern = r"Factual Correctness:.*?(\d+)"
            comprehensiveness_pattern = r"Comprehensiveness:.*?(\d+)"
            clarity_pattern = r"Clarity:.*?(\d+)"
            overall_correctness_pattern = r"Overall Correctness Percentage:.*?(\d+(\.\d+)?)"

            # Find the scores using regex
            factual_correctness_match = re.search(factual_correctness_pattern, response)
            comprehensiveness_match = re.search(comprehensiveness_pattern, response)
            clarity_match = re.search(clarity_pattern, response)
            overall_correctness_match = re.search(overall_correctness_pattern, response)

            # Check if matches were found and extract the values, or default to None if not
            scores = {}
            if factual_correctness_match:
                scores['factual_correctness'] = int(factual_correctness_match.group(1))
            if comprehensiveness_match:
                scores['comprehensiveness'] = int(comprehensiveness_match.group(1))
            if clarity_match:
                scores['clarity'] = int(clarity_match.group(1))
            if overall_correctness_match:
                scores['overall_correctness'] = float(overall_correctness_match.group(1))

            # Return the scores as a JSON object
            return json.dumps(scores, indent=4) 
    

    def evaluate_responses(self):
        """
        Evaluates a set of query-answer pairs from the dataset and prints the evaluation scores.
        """
        # Get the query-answer pairs from the dataset
        qa_pairs = self.data_processor.get_query_and_answer()


        for qa_pair in qa_pairs:
            query = qa_pair.get("input")
            correct_answer = qa_pair.get("answer")

            # Get the generated answer from the RAG model
            generated_answer = self.rag.query(query)
            print("Generated answer \n",generated_answer)

            # Compare the generated answer with the correct answer and get evaluation scores
            evaluation_scores = self.compare_answer(correct_answer, generated_answer, query)
            self.data_processor.save_to_excel(query, correct_answer, generated_answer,evaluation_scores)
            
            print("evaluation_scores",evaluation_scores)
            print(type(evaluation_scores))
            # self.data_processor.save_to_excel(query = query, golden_answer=correct_answer, generated_answer = generated_answer,scores = evaluation_scores )

            # Print the evaluation scores
            print(f"Evaluation scores for query: {query}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Scores: {evaluation_scores}")     

    def start_evaluation(self):
        self.insert_data_in_lightrag()
        self.evaluate_responses()



response_eval = ResponseEvaluator()
response_eval.start_evaluation()



           


        


       
 











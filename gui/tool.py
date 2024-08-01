import time
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


# Load the model
tokenizer = AutoTokenizer.from_pretrained("Eurosmart/bert-qa-mash-covid")
model = AutoModelForQuestionAnswering.from_pretrained("Eurosmart/bert-qa-mash-covid")

# Initialize the question answering pipeline
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

def query(question, context):
    # Use the pipeline to answer the question
    answer = qa_pipeline({
        'context': context,
        'question': question,
    },
    max_answer_len=128,
    max_question_len=2048,
    )
    return answer
	

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)


def get_response(question, context):
    return query(question, context)['answer']
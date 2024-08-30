from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

def create_llm_chain():
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Define the prompt template
    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {question}
    AI: """

    prompt = PromptTemplate(template=template, input_variables=['history', 'question'])

    # Initialize OpenAI LLM
    openai_llm = ChatOpenAI(temperature=0.7, model="gpt-4o", openai_api_key=openai_api_key)

    # Initialize ConversationSummaryMemory
    summary_memory = ConversationSummaryMemory(llm=openai_llm, return_messages=True)

    # Create LLMChain with summary memory
    llm_chain = LLMChain(
        prompt=prompt,
        llm=openai_llm,
        memory=summary_memory,
        verbose=True
    )

    return llm_chain

# def test_llm_chain():
#     llm_chain = create_llm_chain()
    
#     # Define questions inside the function
#     questions = [
#         "What is the capital of France?",
#         "What is the population of Paris?",
#         "Can you summarize our conversation so far?"
#     ]
    
#     for question in questions:
#         print(f"\nHuman: {question}")
#         response = llm_chain({"question": question})
#         print(f"AI: {response['text']}")

# # Call the function to run the test
# test_llm_chain()



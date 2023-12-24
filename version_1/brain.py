from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

def instantiate_chatbot(retriever):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
    qa_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #qa_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
    qa_conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever=retriever, memory=qa_memory)
    return qa_conversation_chain
    
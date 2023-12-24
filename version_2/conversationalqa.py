from operator import itemgetter
from langchain.schema import format_document

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string



class ConversationalQA:
    """This class creates langchain instance that serves as a chat like interface for a question answer application"""
    def __init__(self, llm, retriever):
        """This method creates a chat like QA interface using an llm finetuned for chating and a vector store retreival object"""
        self.base_memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
        self.llm = llm
        self.retriever = retriever 
        self.__conversation_log = []
        self.__question_rephrase_template = """
                                    Given the conversation thread delimited in triple back ticks, carefully rephrase the follow up question\n
                                    into a stand alone question delimited in triple angle brackets, in english.\n
                                    
                                    Chat History:
                                    ```{chat_history}```\n
                                    
                                    Follow up Input: <<<{question}>>>\n
                                    
                                    Standalone question:
                                    """
        self.__answer_template = """
                        Answer the question delimited by triple angle bracktes in english, \n
                        using the context delimited by triple back ticks, don't mention or reference the context information:
                        ```{context}```
                        
                        Question: <<<{question}>>>
                        """

    def __collate_context(self, docs):
        """This is a private method for merging chunks from a pdf relevant to a query
        
        :param docs: A list of documents relevant to a query
        :type List: Documents
        """
        document_ppt = PromptTemplate.from_template(template="{page_content}")
        document_separator="\n\n"
        doc_strings = [format_document(doc, document_ppt) for doc in docs]
        return document_separator.join(doc_strings)
    
    def __standalone_question_component(self):
        """This is a private method that create a langchain component to rephrase conversation history and the most current query into a standalone question"""
        chat_history_lambda = lambda x: get_buffer_string(x["chat_history"])
        question_lambda =  lambda x: x["question"]
        standalone_question_runnable = RunnablePassthrough.assign(chat_history= chat_history_lambda, question = question_lambda)
        question_rephrase_ppt= ChatPromptTemplate.from_template(self.__question_rephrase_template)

        return RunnableParallel(standalone_question= standalone_question_runnable | question_rephrase_ppt | self.llm| StrOutputParser())

    def __system_response(self):
        """This is a private method that creates a langchain component that feeds the 
           rephrased question from __stanalone_question_component() and __colloate_context()
           into a chat llm to generate a response.
        """
        answer_ppt = ChatPromptTemplate.from_template(self.__answer_template)
        final_inputs = {"context": lambda x: self.__collate_context(x["context_docs"]), "question": itemgetter("question")}
        system_response = {"answer": final_inputs | answer_ppt | self.llm, "context_docs": itemgetter("context_docs")}
        return system_response

    def __full_chain(self):
        """This is a private method that puts all the necessary kangchain components together to create 
            the chat qa application.
        """
        relevant_context = {"context_docs": itemgetter("standalone_question") | self.retriever, "question": lambda x: x["standalone_question"]}
        # print('Printing relevant context variable\n')
        # print(relevant_context)
        loaded_conversation_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(self.base_memory.load_memory_variables) | itemgetter("history"))
        complete_chain = loaded_conversation_memory | self.__standalone_question_component() | relevant_context | self.__system_response()
        return complete_chain

    def __chat_history(self):
        """This private method is used to access current chat history between a user and the chat llm
           stored in self.base_memory

           :param self.base_memory: Memory for holding user queries and chat llm responses to the queries
           :type object:ConversationBufferMemory

           :returns: a list of human/chat_llm converstaions for the current session
           :rtype: List
        """
        history =[]
        for entry in self.base_memory.chat_memory.messages:
            entry_dict = entry.dict()
            if entry_dict['example']:
                continue
            else:
                history.append({entry_dict['type']:entry_dict['content']})
        return history
    
    def fully_formed_chain(self):
        """This is a public method that calls the private method __full_chain()
           
           :returns: A custon conversational qa langchain component
        """
        return self.__full_chain()
    
    def query(self, query):
        """This public method invokes the whole qa system"""
        full_chain = self.__full_chain()
        result = full_chain.invoke(query)
        self.base_memory.save_context(query, {"answer": result["answer"].content})
        self.__conversation_log = self.__chat_history()
        return result['answer'].content
        
    def conversation_log(self, query):
        full_chain = self.__full_chain()
        result = full_chain.invoke(query)
        self.base_memory.save_context(query, {"answer": result["answer"].content})
        self.__conversation_log = self.__chat_history()
        return self.__conversation_log
        
    
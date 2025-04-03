import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

#Setup the streamlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text to Math problem solver")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("Please add your GROQ API key to continue")
    st.stop()

llm=ChatGroq(model_name="Gemma2-9b-It",api_key=groq_api_key)

#Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(name="Wikipedia",func=wikipedia_wrapper.run,
                    description="A tool for searching the internet to find the various information on the topics mentioned")

math_chain=LLMMathChain.from_llm(llm=llm,verbose=True)
calculator=Tool(name="Calculator",func=math_chain.run,
                description="A tool for answerig the math related questions. Only mathematical expressions needs to be provided")



prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(input_variables=['question'],template=prompt)

#Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool=Tool(name="reasoning_tool",func=chain.run,
                    description="A tool for answering logic based and reasoning questions")
#Initialize the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi I am Math Chatbot and i can answer your math related question"}]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

#Lets start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generating response.."):
            st.session_state.messages.append({'role':'user','content':question})
            st.chat_message('user').write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write('response:')
            st.success(response)
    else:
        st.warning("Please Enter the question")    


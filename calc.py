# from langchain_groq import ChatGroq
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_community.tools import DuckDuckGoSearchRun

# # 1. LLM Groq (pakai LLaMA 3)
# llm = ChatGroq(
#     model="llama-3.1-8b-instant",   # bisa diganti ke "llama3-8b-8192"
#     temperature=0,
# )

# # 2. Tools
# search = DuckDuckGoSearchRun()

# tools = [
#     Tool(
#         name="Search",
#         func=search.run,
#         description="Useful for answering questions about current events or general knowledge"
#     ),
#     Tool(
#         name="Calculator",
#         func=lambda x: eval(x),
#         description="Useful for math calculations"
#     )
# ]

# # 3. Agent
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True
# )

# # 4. Test Query
# result = agent.run("Cari siapa presiden Indonesia sekarang lalu kalikan umurnya dengan 3.")
# print(result)


import re
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

search = DuckDuckGoSearchRun()

def safe_calculator(x: str):
    try:
        expr = "".join(ch for ch in x if ch.isdigit() or ch in "+-*/. ")
        if expr.strip() == "":
            return "Tidak ada angka yang bisa dihitung."
        return eval(expr)
    except Exception as e:
        return f"Error kalkulasi: {e}"

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Gunakan untuk mencari informasi dari internet."
    ),
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Gunakan untuk melakukan perhitungan matematika sederhana."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
if __name__ == "__main__":
    query = "Cari siapa presiden Indonesia sekarang dan kalikan umurnya dengan 3."
    result = agent.run(query)
    print("\n=== Jawaban Akhir ===")
    print(result)

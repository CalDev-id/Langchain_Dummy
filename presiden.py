import re
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory

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

def reverse_text(x: str):
    return x[::-1]

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Gunakan untuk mencari informasi umum atau berita terbaru."
    ),
    Tool(
        name="Calculator",
        func=safe_calculator,
        description="Gunakan untuk melakukan perhitungan matematika sederhana."
    ),
    Tool(
        name="Reverser",
        func=reverse_text,
        description="Gunakan untuk membalikkan teks yang diberikan."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    print("\n=== Prompt 1 ===")
    q1 = "Siapa presiden Indonesia sekarang?"
    result1 = agent.run(q1)
    print("\nJawaban Akhir 1:", result1)

    print("\n=== Prompt 2 (pakai memory, refer ke 'dia') ===")
    q2 = "Kalikan umurnya dengan 3."
    result2 = agent.run(q2)
    print("\nJawaban Akhir 2:", result2)

    print("\n=== Prompt 3 (multi-tools dalam satu prompt) ===")
    q3 = "Cari siapa presiden Amerika Serikat sekarang, kalikan umurnya dengan 2, lalu balikkan hasilnya."
    result3 = agent.run(q3)
    print("\nJawaban Akhir 3:", result3)

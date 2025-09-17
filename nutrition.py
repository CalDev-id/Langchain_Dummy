import os
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory


# === Load API keys ===
with open("api_key.txt", "r") as f:
    os.environ["GROQ_API_KEY"] = f.read().strip()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "projecttelkom-58f002bf8fa0.json"

# === Connect to Google Sheet ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("projecttelkom-58f002bf8fa0.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url(
    "https://docs.google.com/spreadsheets/d/1j59luEnSt3JfWtIVAI7B_Pa6bSNBuHSjGAVvUSIwWII/edit?usp=sharing"
).sheet1
data = pd.DataFrame(sheet.get_all_records())


# === Helper parser ===
def to_float(x, default=0.0):
    try:
        return float("".join(ch for ch in str(x) if ch.isdigit() or ch == "."))
    except Exception:
        return default

def to_int(x, default=0):
    try:
        return int("".join(ch for ch in str(x) if ch.isdigit()))
    except Exception:
        return default


# === TOOLS ===
def calculate_bmi(input_str: str) -> str:
    """Hitung Body Mass Index (BMI)."""
    parts = input_str.split()
    w = h = 0
    for p in parts:
        if "weight" in p:
            w = to_float(p.split("=")[1])
        if "height" in p:
            h = to_float(p.split("=")[1])
    if w <= 0 or h <= 0:
        return "Format salah. Contoh: weight=70 height=175"
    h_m = h / 100
    bmi = round(w / (h_m * h_m), 2)

    if bmi < 18.5:
        kategori = "Kurus"
    elif 18.5 <= bmi < 24.9:
        kategori = "Normal"
    elif 25 <= bmi < 29.9:
        kategori = "Overweight"
    else:
        kategori = "Obesitas"

    return f"BMI Anda adalah {bmi}. Kategori: {kategori}."


def calculate_bmr(input_str: str) -> str:
    """Hitung Basal Metabolic Rate (BMR)."""
    parts = input_str.split()
    w = h = a = 0
    g = "male"
    for p in parts:
        if "weight" in p:
            w = to_float(p.split("=")[1])
        if "height" in p:
            h = to_float(p.split("=")[1])
        if "age" in p:
            a = to_int(p.split("=")[1])
        if "gender" in p:
            g = p.split("=")[1]
    if w <= 0 or h <= 0 or a <= 0:
        return "Format salah. Contoh: weight=70 height=175 age=25 gender=male"
    if g.lower() == "male":
        bmr = 10 * w + 6.25 * h - 5 * a + 5
    else:
        bmr = 10 * w + 6.25 * h - 5 * a - 161
    return f"BMR Anda adalah {round(bmr, 2)} kalori per hari (energi dasar tubuh)."


def calculate_daily_calories(input_str: str) -> str:
    """Hitung kebutuhan kalori harian."""
    parts = input_str.split()
    b = 0
    act = "sedentary"
    for p in parts:
        if "bmr" in p:
            b = to_float(p.split("=")[1])
        if "activity" in p:
            act = p.split("=")[1]
    factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    kal = round(b * factors.get(act, 1.2), 2)
    return f"Kebutuhan kalori harian Anda sekitar {kal} kalori berdasarkan aktivitas {act}."


def calculate_macros(input_str: str) -> str:
    """Hitung makronutrien dari kalori."""
    c = to_float(input_str.split("=")[1])
    protein = round((c * 0.3) / 4, 1)
    fat = round((c * 0.25) / 9, 1)
    carbs = round((c * 0.45) / 4, 1)
    return (
        f"Dari {c} kalori: Protein {protein}g, Lemak {fat}g, Karbohidrat {carbs}g."
    )


def lookup_food(input_str: str) -> str:
    """Cari nutrisi makanan dari Google Sheet."""
    food_name = input_str.split("=")[1]
    match = data[data["name"].str.lower() == food_name.lower()]
    if match.empty:
        return "Makanan tidak ditemukan."
    row = match.to_dict("records")[0]
    return f"{row['name']}: {row['calories']} kalori, Protein {row['proteins']}g, Lemak {row['fat']}g, Karbohidrat {row['carbohydrate']}g."


def compare_foods(input_str: str) -> str:
    """Bandingkan 2 makanan dari Google Sheet."""
    parts = input_str.split("food2=")
    f1 = parts[0].split("=")[1].strip()
    f2 = parts[1].strip()
    df1 = data[data["name"].str.lower() == f1.lower()]
    df2 = data[data["name"].str.lower() == f2.lower()]
    if df1.empty or df2.empty:
        return "Salah satu makanan tidak ditemukan."
    d1, d2 = df1.to_dict("records")[0], df2.to_dict("records")[0]
    return (
        f"{d1['name']} → {d1['calories']} kalori, Protein {d1['proteins']}g, Lemak {d1['fat']}g, Karbo {d1['carbohydrate']}g.\n"
        f"{d2['name']} → {d2['calories']} kalori, Protein {d2['proteins']}g, Lemak {d2['fat']}g, Karbo {d2['carbohydrate']}g."
    )


# === Initialize LLM ===
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# === Register tools ===
tools = [
    Tool.from_function(func=calculate_bmi, name="calculate_bmi", description="Hitung BMI. Input format: 'weight=70 height=175'"),
    Tool.from_function(func=calculate_bmr, name="calculate_bmr", description="Hitung BMR. Input format: 'weight=70 height=175 age=25 gender=male'"),
    Tool.from_function(func=calculate_daily_calories, name="calculate_daily_calories", description="Hitung kebutuhan kalori harian. Input format: 'bmr=1700 activity=moderate'"),
    Tool.from_function(func=calculate_macros, name="calculate_macros", description="Hitung makronutrien dari kalori. Input format: 'calories=2500'"),
    Tool.from_function(func=lookup_food, name="lookup_food", description="Cari nutrisi makanan dari Google Sheet. Input format: 'name=Abon'"),
    Tool.from_function(func=compare_foods, name="compare_foods", description="Bandingkan nutrisi 2 makanan. Input format: 'food1=Abon food2=Abon haruwan'"),
]


# === Build Agent ===
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False,
#     agent_kwargs={
#         "system_message": SystemMessage(
#             content="Kamu adalah asisten nutrisi yang selalu menjawab dalam bahasa Indonesia. "
#                     "Berikan hasil yang jelas, ramah, dan sertakan sedikit penjelasan agar mudah dipahami."
#         )
#     }
# )
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    agent_kwargs={
        "system_message": SystemMessage(
            content=(
                "Kamu adalah asisten nutrisi dalam bahasa Indonesia. "
                "Jika pengguna meminta perhitungan (BMI, BMR, kalori, makro, dll) "
                "tapi belum memberi semua data yang dibutuhkan (seperti berat, tinggi, umur, atau gender), "
                "jangan langsung hitung. Sebaliknya, tanyakan dulu data yang kurang "
                "dengan ramah. "
                "Jika semua data sudah lengkap, barulah panggil tool yang sesuai."
            )
        )
    }
)



# === Contoh interaksi ===
if __name__ == "__main__":
    # print(agent.run("Hitung BMI saya berat 70kg tinggi 175cm"))
    # print(agent.run("Saya laki-laki umur 25, berat 70kg, tinggi 175cm. Hitung BMR saya"))
    # print(agent.run("Kalau BMR saya 1700 dan aktivitas moderate, berapa kebutuhan kalori harian saya?"))
    # print(agent.run("Hitung makronutrien dari 2500 kalori"))
    # print(agent.run("Cari tahu nutrisi Abon"))
    # print(agent.run("Bandingkan nutrisi Abon dan Abon haruwan"))
    print(agent.run("apakah kamu bisa hitung bmi saya?"))
    # print(agent.run("Hitung BMI saya berat 54kg tinggi 169cm"))

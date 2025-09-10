from dotenv import load_dotenv              # .env dosyasındaki API key gibi ortam değişkenlerini yüklemek için
from pydantic import BaseModel              # Veri şeması/validasyonu için Pydantic sınıfı
from langchain_openai import ChatOpenAI     # OpenAI modellerine (gpt-4, gpt-4o-mini vs) bağlanmak için
from langchain_anthropic import ChatAnthropic  # Anthropic modellerine (Claude) bağlanmak için
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini modellerine bağlanmak için
from langchain_core.prompts import ChatPromptTemplate      # Promptları (system/human mesajları) şablonlamak için
from langchain_core.output_parsers import PydanticOutputParser  # LLM çıktısını Pydantic şemasına göre parse etmek için
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()  # Proje kökünde .env dosyasını okuyarak OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY gibi değerleri yükler

class ResearchResponse(BaseModel):   # Pydantic ile bir çıktı şeması tanımlıyoruz
    research: str                    # "research" alanı → string (detaylı araştırma kısmı)
    summary: str                     # "summary" alanı → string (kısa özet)
    sources: list[str]               # "sources" alanı → liste (kaynak linkleri veya referanslar)
    tools_used: list[str]            # "tools_used" alanı → liste (hangi araçlar/model kullanıldı bilgisi)


# llm = ChatOpenAI(model="gpt-4o-mini")               # OpenAI’nin GPT-4o-mini modelini kullanmak için (yorum satırı)
# llm2 = ChatAnthropic(model="claude-3-5-sonnet-20240620")   # Anthropic Claude modelini kullanmak için (yorum satırı)
llm3 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")     # Google Gemini 1.5 Flash modelini seçiyoruz (aktif kullanılan)

# response = llm3.invoke("What is the meaning of life?")    # Basit bir test sorgusu (yorum satırı)
# print(response)                                           # Çıktıyı yazdırmak için (yorum satırı)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)  
# LLM çıktısını otomatik olarak ResearchResponse şemasına parse edecek parser.
# Yani modelin ürettiği JSON çıktısını alıp Pydantic objesine dönüştürür.

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),   # Sistem mesajı: modele rolünü tanımlıyoruz (araştırma asistanı) + çıktıyı parser formatında vermesini söylüyoruz
        ("placeholder", "{chat_history}"),    # Önceki konuşmalar (history) buraya otomatik doldurulacak
        ("human", "{query}"),                 # Kullanıcının sorusu (query) buraya gelecek
        ("placeholder", "{agent_scratchpad}"),# Agent çalışma alanı (kullandığı araçların intermediate çıktıları buraya gelir)
    ]
).partial(format_instructions=parser.get_format_instructions())
# Prompt template’e parser’ın beklediği format talimatlarını ekliyoruz.
# Böylece model çıktısını JSON/Pydantic şemasıyla uyumlu üretmeye zorlarız.
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm3, 
    prompt=prompt, 
    tools=tools
    )
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your query: ")
response = agent_executor.invoke({"query": query})
try:
    st_response = parser.parse(response["output"])
    print(st_response)
except Exception as e:
    print("Error parsing response: ", e,"Raw response: ", response)
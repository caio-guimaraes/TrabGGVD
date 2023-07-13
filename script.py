from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array, concat_ws, lit
from pyspark.sql.types import ArrayType, StringType, StructType, StructField
from empath import Empath
from operator import add
import os
from sys import argv

# Função para tokenizar a entrada
def separa(line):
    result = []

    if isinstance(line, str):
        result = line.lower().split(" ")

    return result

# Função para processar as emoções usando Empath
def process_emotions(top_words):
    if len(top_words) == 0:
        return ""

    lexicon = Empath()
    emotions = lexicon.analyze(top_words, normalize=True)
    sorted_emotions = sorted(emotions.items(), key=lambda x: -x[1])
    filtered_emotions = [(emotion, score) for emotion, score in sorted_emotions if score > 0.0500]
    formatted_emotions = [f"{emotion} {score}" for emotion, score in filtered_emotions]
    return ", ".join(formatted_emotions)

# Função para extrair o nome do artista do arquivo
def extract_artist(file_path):
    file_name = os.path.basename(file_path)
    return os.path.splitext(file_name)[0]

# Diretório contendo os arquivos CSV
diretorio = argv[1]

# Sessão do Spark
spark = SparkSession.builder.appName("PythonWordCount").getOrCreate()

# Carrega uma lista de stopwords
with open("gist_stopwords.txt", "r") as gist_file:
    content = gist_file.read()
    stop_words = content.split(",")
    stop_words.append("")

# Define as funções UDF para uso com o DataFrame do Spark
separa_udf = udf(separa, ArrayType(StringType()))
process_emotions_udf = udf(process_emotions, StringType())
extract_artist_udf = udf(extract_artist, StringType())

# Lista de arquivos CSV no diretório
arquivos_csv = [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio) if arquivo.endswith(".csv")]

# Verifica se existem arquivos CSV no diretório
if len(arquivos_csv) == 0:
    print("Nenhum arquivo CSV encontrado no diretório.")
    spark.stop()
    exit()

# Define a estrutura de colunas esperada
schema = StructType([
    StructField("Artist", StringType(), nullable=True),
    StructField("Emotions", StringType(), nullable=True)
])

# Processa os dados para cada arquivo CSV
output_data = []

for arquivo_csv in arquivos_csv:
    if os.path.isfile(arquivo_csv):
        df = spark.read.format("csv").options(header="true", inferSchema="true").load("file://" + os.path.abspath(arquivo_csv))
        nome_artista = extract_artist(arquivo_csv)
        df = df.withColumn("Artist", array(lit(nome_artista)))
        df = df.withColumn("Lyric", separa_udf(df["Lyric"]))

        mapped_rdd = df.select("Lyric").rdd.flatMap(lambda line: line.Lyric).filter(lambda x: x not in stop_words).map(lambda word: (word, 1))
        reduced_rdd = mapped_rdd.reduceByKey(add)
        top20 = reduced_rdd.takeOrdered(20, key=lambda x: -x[1])
        top_words = [key for (key, value) in top20]
        emotions = process_emotions(top_words)  # Executa a função diretamente

        output_data.append((nome_artista, emotions))
    else:
        print(f"Arquivo CSV não encontrado: {arquivo_csv}")



# Cria um RDD com os dados de saída
output_rdd = spark.sparkContext.parallelize(output_data)

# Cria um DataFrame com os dados de saída
output_df = spark.createDataFrame(output_rdd, schema=schema)

# Combina as colunas "Artist" e "Emotions" em uma única coluna "Output" com os valores de emoção
output_df = output_df.withColumn("Output", concat_ws(" - ", col("Artist"), col("Emotions")))

# Filtra apenas os sentimentos com valor maior que 0.0500
output_df = output_df.filter(output_df["Emotions"] != "")

# Salva os resultados em um arquivo de texto
output_file = "output"
output_path = os.path.join(os.getcwd(), output_file)
output_df.select("Output").coalesce(1).write.text(output_path)

spark.stop()

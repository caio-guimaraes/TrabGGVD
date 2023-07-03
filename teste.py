from empath import Empath
from operator import add
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
import pyspark.sql.functions as f
from sys import argv

def separa(line):
    result = []
    if isinstance(line.Lyric, str):
        result = line.Lyric.lower().split(" ")
    return result

arq = argv[1]

spark = SparkSession\
  .builder \
  .appName("PythonWordCount") \
  .getOrCreate()

gist_file = open("gist_stopwords.txt", "r")
try:
    content = gist_file.read()
    stop_words = content.split(",")
    stop_words.append("")
finally:
    gist_file.close()

out = open("file.txt", "w")
data = spark.read.format('csv').options(header='true', inferSchema='true') \
.load(arq).select("Lyric")
# faz o mapping de todos os rdds
mapped_rdd = data.rdd.flatMap (lambda line: separa(line)).filter(lambda x: x not in stop_words)\
    .map(lambda word: (word, 1)).reduceByKey (add)
word_count = mapped_rdd.takeOrdered(20, key=lambda x: -x[1])

top_words = set().union((key for (key, value) in word_count))

lexicon = Empath()

my_dict = lexicon.analyze(list(top_words), normalize=True)
for key, value in my_dict.items():
    if value > 0:
        out.write("%s: %f\n"%(key, value))

out.close()

spark.stop()

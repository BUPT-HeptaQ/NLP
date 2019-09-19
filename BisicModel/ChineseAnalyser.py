# only the original prototype
from __future__ import unicode_literals
import sys,os
import jieba
import jieba.analyse as analyse
sys.path.append("../")
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser

analyzer = jieba.analyse.ChineseAnalyzer()
schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True), analyzer=analyzer)

if not os.path.exists("tmp"):
    os.mkdir("tmp")

ix = create_in("tmp", schema)
writer = ix.writer()
# add different docs
writer.add_document(
    title="document1", path="/a", content="This is the first document we've added!"
)

writer.add_document(
    title="document2", path="/b", content="The second one 用来测试中文吧 is even more interesting!"
)

writer.commit()
searcher = ix.searcher()
parser = QueryParser("content", schema=ix.schema)

for keywords in ("你", "first", "中文"):
    print(keywords + "results are as following: ")
    q = parser.parse(keywords)
    results = searcher.search(q)
    for hit in results:
        print(hit.highlights("content"))
    print("\n------------cut line-------------\n")

for t in analyzer(""):
    print(t.text)


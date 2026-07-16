import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from graph.nodes import _call
from agents.clause_rag import ClauseRAG
from agents.verdict_rag import VerdictRAG

model = SentenceTransformer('BAAI/bge-m3')
cr = ClauseRAG(model, _call)
vr = VerdictRAG(model, _call)

print('=== 약관 RAG ===')
r = cr.answer('비보호 좌회전 사고 면책 여부')
print(r['answer'])
print('근거 조항:', r['articles'])
print(r['answer'])
print('근거 조항:', r['articles'])

print()
print()
print('=== 판례 RAG ===')
r = vr.answer('비보호 좌회전 직진차 과실비율')
print(r['answer'])
print('과실비율:', r['fault_ratio'])  
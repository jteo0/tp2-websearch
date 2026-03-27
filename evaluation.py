import re
from bsbi import BSBIIndex
from compression import VBEPostings
import math

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
    score = 0.
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] / math.log2(i + 1)
    return score

def ndcg(ranking):
    ideal = sorted(ranking, reverse=True)
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 0.
    return dcg(ranking) / ideal_score

def ap(ranking):
    score = 0.
    num_relevant = 0
    for i in range(1, len(ranking) + 1):
        if ranking[i - 1] == 1:
            num_relevant += 1
            score += num_relevant / i
    if num_relevant == 0:
        return 0.
    return score / num_relevant

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores_tfidf, dcg_scores_tfidf, ndcg_scores_tfidf, ap_scores_tfidf = [], [], [], []
    rbp_scores_bm25, dcg_scores_bm25, ndcg_scores_bm25, ap_scores_bm25 = [], [], [], []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      # TF-IDF
      ranking_tfidf = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_tfidf.append(qrels[qid][did])
      rbp_scores_tfidf.append(rbp(ranking_tfidf))
      dcg_scores_tfidf.append(dcg(ranking_tfidf))
      ndcg_scores_tfidf.append(ndcg(ranking_tfidf))
      ap_scores_tfidf.append(ap(ranking_tfidf))
      
      # BM25
      ranking_bm25 = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k=k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_bm25.append(qrels[qid][did])
      rbp_scores_bm25.append(rbp(ranking_bm25))
      dcg_scores_bm25.append(dcg(ranking_bm25))
      ndcg_scores_bm25.append(ndcg(ranking_bm25))
      ap_scores_bm25.append(ap(ranking_bm25))

  n = len(rbp_scores_tfidf)
  print("Hasil evaluasi TF-IDF terhadap 30 queries")
  print("RBP score  =", sum(rbp_scores_tfidf) / n)
  print("DCG score  =", sum(dcg_scores_tfidf) / n)
  print("NDCG score =", sum(ndcg_scores_tfidf) / n)
  print("MAP score  =", sum(ap_scores_tfidf) / n)

  print("\nHasil evaluasi BM25 terhadap 30 queries")
  print("RBP score  =", sum(rbp_scores_bm25) / n)
  print("DCG score  =", sum(dcg_scores_bm25) / n)
  print("NDCG score =", sum(ndcg_scores_bm25) / n)
  print("MAP score  =", sum(ap_scores_bm25) / n)

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)

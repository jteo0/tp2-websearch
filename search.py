from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print(f"Query: {query}")
    print(f"{'Rank':<5} {'TF-IDF Score':>14} {'BM25 Score':>12}  Dokumen")
    print("-" * 70)

    tfidf_results = BSBI_instance.retrieve_tfidf(query, k=10)
    bm25_results  = BSBI_instance.retrieve_bm25(query, k=10)

    for i, ((s1, d1), (s2, d2)) in enumerate(zip(tfidf_results, bm25_results), 1):
        print(f"{i:<5} {s1:>14.3f} {s2:>12.3f}  {d1}")
    print()
import os
import pickle
import contextlib
import heapq
import time
import math
import sys

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs
    
    def parse_block_spimi(self, block_dir_relative):
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((token, self.doc_id_map[docname]))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def invert_write_spimi(self, td_pairs, index):
        local_dict = {}  # term_string -> {doc_id: tf}
        for term_str, doc_id in td_pairs:
            if term_str not in local_dict:
                local_dict[term_str] = {}
            local_dict[term_str][doc_id] = local_dict[term_str].get(doc_id, 0) + 1

        for term_str in sorted(local_dict.keys()):
            term_id = self.term_id_map[term_str]
            sorted_doc_ids = sorted(local_dict[term_str].keys())
            tf_list = [local_dict[term_str][doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, sorted_doc_ids, tf_list)
    
    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked retrieval dengan skema TaaT menggunakan BM25.

        Parameters
        ----------
        query : str
            Query tokens dipisahkan spasi
        k : int
            Jumlah dokumen teratas yang dikembalikan
        k1 : float
            Parameter saturasi TF (default 1.2)
        b : float
            Parameter normalisasi panjang dokumen (default 0.75)

        Returns
        -------
        List[(float, str)]
            Top-K dokumen terurut menurun berdasarkan skor BM25
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()
                if word in self.term_id_map.str_to_id]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N

            term_data = {}
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue
                pos, df, len_post, len_tf, max_tf = merged_index.postings_dict[term]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf_norm_upper = (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b + b * 1 / avgdl))
                upper_bound = idf * tf_norm_upper
                postings, tf_list = merged_index.get_postings_list(term)
                term_data[term] = {
                    'postings': postings,
                    'tf_list': tf_list,
                    'idf': idf,
                    'upper_bound': upper_bound,
                    'ptr': 0
                }

            if not term_data:
                return []

            def get_current_doc(term):
                ptr = term_data[term]['ptr']
                postings = term_data[term]['postings']
                return postings[ptr] if ptr < len(postings) else float('inf')

            def advance_to(term, target):
                postings = term_data[term]['postings']
                ptr = term_data[term]['ptr']
                while ptr < len(postings) and postings[ptr] < target:
                    ptr += 1
                term_data[term]['ptr'] = ptr

            heap = []
            active_terms = list(term_data.keys())

            while True:
                active_terms = [t for t in active_terms if get_current_doc(t) != float('inf')]
                if not active_terms:
                    break

                active_terms.sort(key=lambda t: get_current_doc(t))

                threshold = heap[0][0] if len(heap) == k else 0
                cumulative = 0
                pivot_doc = None
                for term in active_terms:
                    cumulative += term_data[term]['upper_bound']
                    if cumulative > threshold:
                        pivot_doc = get_current_doc(term)
                        break

                if pivot_doc is None:
                    break

                first_doc = get_current_doc(active_terms[0])
                if first_doc == pivot_doc:
                    score = 0
                    dl = merged_index.doc_length.get(pivot_doc, avgdl)
                    for term in active_terms:
                        if get_current_doc(term) == pivot_doc:
                            ptr = term_data[term]['ptr']
                            tf = term_data[term]['tf_list'][ptr]
                            idf = term_data[term]['idf']
                            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                            score += idf * tf_norm
                            term_data[term]['ptr'] += 1

                    if len(heap) < k:
                        heapq.heappush(heap, (score, pivot_doc))
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, (score, pivot_doc))
                else:
                    for term in active_terms:
                        if get_current_doc(term) < pivot_doc:
                            advance_to(term, pivot_doc)

        docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in heap]
        return sorted(docs, key=lambda x: x[0], reverse=True)

    def index(self, method='bsbi'):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            if method == 'spimi':
                td_pairs = self.parse_block_spimi(block_dir_relative)
            else:
                td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                if method == 'spimi':
                    self.invert_write_spimi(td_pairs, index)
                else:
                    self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    encoding = VBEPostings if len(sys.argv) < 2 or sys.argv[1] == "vbe" else EliasGammaPostings
    method = 'spimi' if len(sys.argv) > 2 and sys.argv[2] == 'spimi' else 'bsbi'
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=encoding,
                              output_dir='index')
    BSBI_instance.index(method=method)

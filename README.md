# TP2 - Search Engine "from Scratch"
## About

This project was developed as a personal exploration on search engines and how they work for a Web Search and Information Retrieval class. It is a bare-bones 'from Scratch' search engine that implements an information retrieval system that uses BSBI indexing, multiple compression schemes, supports ranked retrieval via TF-IDF and BM25, and evaluates retrievals using the RBP, DCG, NDCG, and MAP metrics.

## Features

- Efficiently create inverted indexes with BSBI (Block-Sort Based Indexing)
- Compare the effectiveness of VBE and Elias-Gamma compression schemes
- Compare the effectiveness of implementing TF-IDF and BM25

## How does it work?
After cloning or pulling the project, install tqdm. You can use the following command:
```
python install tqdm
```

Once that's finished installing, ```bsbi.py``` is run first to build the index. You can use either VBE or Elias-Gamma here by adjusting the command you use. Whenever you use a different compression algorithm or adjust anything in the collection folder, you have to empty the index folder and rerun ```bsbi.py```. Otherwise, this only needs to be run once.
```
python bsbi.py vbe
# any other input will use Elias-Gamma
```

Then, run ```search.py``` to try the sample queries. Queries can be adjusted by directly editing the query variable within ```search.py```.
```
python search.py
```

You can evaluate the quality of the retrieval using ```evaluation.py```, which will compare the retrievals that use TF-IDF and BM25 against each other.
```
python evaluation.py
```

Additionally, 

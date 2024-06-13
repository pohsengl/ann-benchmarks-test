import chromadb
import numpy
from ..base.module import BaseANN
import subprocess
import sys
from pathlib import Path
import time
import threading

class ChromaBase(BaseANN):

    dir_path = {
        "total": str(Path.home() / "chromadb_data")
    }

    def __init__(self, metric) -> None:

        supported_metrics = {
            "angular": "cosine",
            "euclidean": "l2",
        }
        if metric not in supported_metrics:
            raise NotImplementedError(f"{metric} is not implemented")
        
        distance_metric = supported_metrics[metric]
    
        # start chromadb
        chroma_port = 8010

        def _start_chroma(port):
            subprocess.run(f"chroma run --path {self.dir_path['total']} --port {port}", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

        # create a thread
        self._chroma_thread = threading.Thread(target=_start_chroma, args=(chroma_port,))
        self._chroma_thread.daemon = True
        self._chroma_thread.start()

        time.sleep(5)

        self._client = chromadb.HttpClient(host="localhost", port=chroma_port)
        self._collection = self._client.create_collection(name="benchmark", metadata={"hnsw:space": distance_metric})
        print("Chroma collection created")

    def fit(self, X) -> None:
        print("Fitting Chroma with data...")
        def divide_into_batches(arr, batch_size):
            for i in range(0, len(arr), batch_size):
                yield arr[i:i + batch_size]

        CHROMA_BATCH_LIMIT = 40000
        chroma_batches = divide_into_batches(X, CHROMA_BATCH_LIMIT)

        item_id = 0
        for batch in chroma_batches:
            print(f"Adding batch {item_id} to {item_id + len(batch)}, batch shape: {batch.shape}...")
            self._collection.add(
                embeddings=batch,
                ids=[str(i) for i in range(item_id, item_id + len(batch))]
            )
            item_id += len(batch)

        print("Chroma data added")

    def query(self, q: numpy.array, n: int) -> numpy.array:
        print(f"Querying for {n} nearest neighbors...")
        query_result = self._collection.query(
            query_embeddings=[q.tolist()],
            n_results=n
        )

        #print(query_result)

        # by default, it returns ids, "metadatas", "documents", "distances"
        return numpy.array([int(i) for i in query_result['ids'][0]])

    def set_query_arguments(self):
        pass

    def __str__(self) -> str:
        return "ChromaBase"
    
if __name__ == "__main__":
    chroma_inst = ChromaBase()

    # test fit
    train_set = numpy.random.rand(100, 3)
    test_set = numpy.random.rand(10, 3)
    chroma_inst.fit(train_set)

    # test query
    query = test_set[0]
    n = 5
    result = chroma_inst.query(query, n)
    print(result)

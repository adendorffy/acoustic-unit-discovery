
from collections import defaultdict

class Cluster:
    def __init__(self,id, word_dict=None, true_words=None):
        self.id = id
        self.length = len(word_dict) if word_dict else 0
        self.word_dict = word_dict if word_dict is not None else []
        self.true_word_dict = true_words if true_words is not None else []
    
    def add_word_unit(self, id, index, file):
        word_unit = WordUnit(file, index, id)
        self.length += 1
        self.word_dict.append(word_unit)

    def add_true_word(self, word):
        self.true_word_dict.append(word)

    @classmethod
    def print_cluster(self, cluster):
        print(f"Cluster {cluster.id}")
        for word in cluster.word_dict:
            print(f"Word {word.id}: Index {word.index} in File {word.file}")
    
    def cluster_purity(self):

        word_counts = {}
        for word in self.true_word_dict:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_count = max(word_counts.values()) if word_counts else 0
        cluster_purity = max_count / self.length if self.length > 0 else 0

        self.purity = cluster_purity

    @classmethod
    def duplicate_clusters(self, clusters):
        cluster_dict = defaultdict(int)

        for cluster in clusters:
            cluster_set = frozenset(cluster)  
            cluster_dict[cluster_set] += 1  

        duplicate_count = sum(1 for count in cluster_dict.values() if count > 1)

        return duplicate_count

class WordUnit:
    def __init__(self, file, index, id):
        self.index = int(index)
        self.file = file
        self.id = int(id)
import cppyy

cppyy.cppdef(r"""
#include <vector>
#include <memory>
#include <queue>
#include <string_view>
#include <string>
#include <unordered_map>
#include <utility>

void add_vocab(std::unordered_map<int, std::string>& m, int k, const std::string& v) {
    m[k] = v;
}

void add_bytes_to_id(std::unordered_map<std::string, int>& m, const std::string& k, int v) {
    m[k] = v;
}

void add_merge(std::vector<std::pair<std::string, std::string>>& merges, const std::string& a, const std::string& b) {
    merges.push_back({a, b});
}

class Node {
public:
    int value;
    Node* prev;
    Node* next;
    int pos; // for tie-breaking in the heap

    explicit Node(int val) : value(val), prev(nullptr), next(nullptr), pos(-1) {}

    Node(int val, Node* prev, Node* next, int pos) : value(val), prev(prev), next(next), pos(pos) {}
};

struct HeapEntry {
    int rank;
    int tie_breaker;
    Node *node;

    bool operator<(const HeapEntry& other) const {
        if (this->rank == other.rank) {
            return this->tie_breaker > other.tie_breaker;
        }
        return this->rank > other.rank;
    }
};

class BPEDoubplyLinkedList {
public: 
    Node sentinel{0};
    std::vector<std::unique_ptr<Node>> nodes; // To manage the lifetime of nodes
    int next_pos = 0; // To assign positions to nodes for tie-breaking in the heap

    BPEDoubplyLinkedList(){
        sentinel.prev = &sentinel;
        sentinel.next = &sentinel;
    }

    BPEDoubplyLinkedList(const std::vector<int>& token_ids) {
        sentinel.prev = &sentinel;
        sentinel.next = &sentinel;

        for (auto x : token_ids) {
            this->append(x);
        }
    }

    void append(int val) {
        this->insert_between(val, sentinel.prev, &sentinel);
    }

    Node* insert_between(int val, Node* left, Node* right) {
        nodes.emplace_back(std::make_unique<Node>(val, left, right, next_pos++));
        Node* new_node = nodes.back().get();
        left->next = new_node;
        right->prev = new_node;
        return new_node;
    }

    void merge_with_next(Node* left, int new_val) {
        left->value = new_val;
        this->unlink(left->next);
    }

    void unlink(Node* node) {
        if (node == nullptr) return;
        node->prev->next = node->next;
        node->next->prev = node->prev;
        node->prev = nullptr;
        node->next = nullptr;
    }
};

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        // std::hash<T>{} calls the constructor of std::hash<T>
        // (p.first) calls size_t operator() of std::hash class 
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

std::unordered_map<std::pair<int, int>, int, PairHash> build_rank(
    const std::vector<std::pair<std::string, std::string> >& merges,
    const std::unordered_map<std::string, int>& bytes_to_ids) 
{   
    // rank is to map adacent token_ids to their merge order.
    // the small the order/rank, the more frequently they appear in the training data
    // we need the rank to determine which pair to merge first
    // rank will be put in a minheap
    std::unordered_map<std::pair<int, int>, int, PairHash> rank;
    int n = merges.size();
    for (int i = 0; i < n; i++) {
        const auto& merge = merges[i];
        // prefer to use .at instead of [] for good practice here
        // [key] will insert a new key with default value even though the key is not there
        // since [key] modify the map, it cannot work with a const map.
        // .at will throw an exception but [key] will not. 
        int id1 = bytes_to_ids.at(merge.first);
        int id2 = bytes_to_ids.at(merge.second);
        rank[{id1, id2}] = i;
    }
    return rank;
}

// cppyy will convert bytes in python to std::string_view in C++. We can use this to our advantage to avoid unnecessary copying of strings.
// this convert is zero_copy
// python has vocab: dict[int, bytes] -> cpp 的 std::string是处理任意二进制数据（包含 \0）的标准容器，cppyy对此转换有支持
std::vector<int> encode_single(
    std::string_view text,
    const std::unordered_map<int, std::string>& vocab,
    const std::unordered_map<std::string, int>& bytes_to_ids,
    const std::unordered_map<std::pair<int, int>, int, PairHash>& rank
) {
    if (text.empty()) {
        return {};
    }

    std::vector<int> token_ids;
    token_ids.reserve(text.size());

    for (char c : text) {
        int byte_value = static_cast<unsigned char>(c);
        token_ids.push_back(bytes_to_ids.at(std::string(1, byte_value)));
    }

    BPEDoubplyLinkedList DLL(token_ids);
    
    // (int rank, int tie_breaker, Node* node)
    std::priority_queue<HeapEntry> heap;   

    // passed left node
    auto try_push = [&] (Node* node) -> void {
        if (node == nullptr || node == &DLL.sentinel) {
            return;
        }
        
        Node* next_node = node->next;
        if (next_node == nullptr || next_node == &DLL.sentinel) {
            return;
        }
        
        auto it = rank.find({node->value, next_node->value});
        if (it != rank.end()) {
            heap.push(HeapEntry{it->second, node->pos, node});
        }
    };

    // initialize the heap with all adjacent pairs
    for (Node* cur = DLL.sentinel.next; cur != &DLL.sentinel; cur = cur->next) {
        try_push(cur);
    }

    while (!heap.empty()) {
        auto entry = heap.top();
        int cur_rank = entry.rank;
        Node* node = entry.node;
        heap.pop();

        if (node->prev == nullptr || node->next == nullptr) {
            continue;
        }
        Node* next_node = node->next;
        if (next_node == &DLL.sentinel) {
            continue;
        }
        // cannot use rank.at({node->value, next_node->value}) bc map.at will throw an exception if the key is not found, but here we just want to skip unfound keys
        auto it = rank.find({node->value, next_node->value});
        if (it == rank.end() || it->second != cur_rank) {
            continue;
        }

        std::string merged_bytes = vocab.at(node->value) + vocab.at(next_node->value);
        int new_id = bytes_to_ids.at(merged_bytes);
        DLL.merge_with_next(node, new_id);
        if (node->prev != &DLL.sentinel) {
            try_push(node->prev);
        }
        if (node->next != &DLL.sentinel) {
            try_push(node);
        }
    }

    std::vector<int> out;
    for (Node* cur = DLL.sentinel.next; cur != &DLL.sentinel; cur = cur->next) {
        out.push_back(cur->value);
    }
    return out;
}

std::vector<int> encode_words(
    const std::vector<std::string>& words,
    const std::unordered_map<int, std::string>& vocab,
    const std::unordered_map<std::string, int>& bytes_to_ids,
    const std::unordered_map<std::pair<int, int>, int, PairHash>& rank
) 
{
    if (words.empty()) return {};

    std::vector<int> out;
    for (const auto& word : words) {
        auto res = encode_single(word, vocab, bytes_to_ids, rank);
        out.insert(out.end(), res.begin(), res.end());
    }
    return out;
}
             
std::vector<int> encode_words_parallel(
    const std::vector<std::string>& words,
    const std::unordered_map<int, std::string>& vocab,
    const std::unordered_map<std::string, int>& bytes_to_ids,
    const std::unordered_map<std::pair<int, int>, int, PairHash>& rank
) {
    if (words.empty()) return {};
    int num_threads = std::thread::hardware_concurrency();
    int n = words.size();
    int chunk_size = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> thread_pool(num_threads);
    std::vector<std::vector<int>> results(n); // to store results from each thread

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size; // reach chunk is a list of std::string (words)
        int end = std::min(start + chunk_size, n);
        thread_pool.emplace_back([start, end, &words, &vocab, &bytes_to_ids, &rank, &results](){
            for (int j = start; j < end; j++) {
                results[j] = encode_single(words[j], vocab, bytes_to_ids, rank);
            }  
        });
    }
    
    for (auto& thread : thread_pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    std::vector<int> out;
    for (const auto& res : results) {
        out.insert(out.end(), res.begin(), res.end());
    }
    return out;
}

""");
import regex as re
import json
from typing import Iterator

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # Convert complex byte dicts into cppyy compatible std::unordered_map
        self._cpp_vocab = cppyy.gbl.std.unordered_map['int', 'std::string']()
        for k, v in self.vocab.items():
            cppyy.gbl.add_vocab(self._cpp_vocab, k, v)

        self._cpp_bytes_to_id = cppyy.gbl.std.unordered_map['std::string', 'int']()
        for k, v in self.bytes_to_id.items():
            cppyy.gbl.add_bytes_to_id(self._cpp_bytes_to_id, k, v)

        
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_patterns = re.compile(
                "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            )
            self.special_to_id = {
                tok: self.bytes_to_id[tok.encode("utf-8")]
                for tok in self.special_tokens
            }
        else:
            self.special_patterns = None
            self.special_to_id = {}

        merges_cpp = cppyy.gbl.std.vector['std::pair<std::string, std::string>']()
        for m in self.merges:
            cppyy.gbl.add_merge(merges_cpp, m[0], m[1])
        self.rank = cppyy.gbl.build_rank(merges_cpp, self._cpp_bytes_to_id)

    @classmethod
    def from_files(cls, vocab_filepath: str,  
                   merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
        with open(vocab_filepath, "rb") as f:
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            for line in f:
                cleaned = line.rstrip()
                if cleaned and len(cleaned.split(b" ")) == 2:
                    a, b = cleaned.split(b" ")
                    merges.append((a, b))
        return cls({int(k): v.encode("utf-8") if isinstance(v, str) else v for k, v in vocab.items()}, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_patterns:
            pieces = self.special_patterns.split(text)
        else:
            pieces = [text]
            
        out = []
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens:
                out.append(self.special_to_id[piece])
                continue

            raw_words = []
            for m in PAT.finditer(piece):
                raw = m.group(0).encode("utf-8")
                raw_words.append(raw)
                # out.extend(cppyy.gbl.encode_single(raw, self._cpp_vocab, self._cpp_bytes_to_id, self.rank))
            # 上面如果逐个word调用cppyy.global_name_space.encode_single，速度会暴跌，原因是多次在cpp和python之间切换会有大量的call overhead。
            if raw_words:
                out.extend(cppyy.gbl.encode_words_parallel(raw_words, self._cpp_vocab, self._cpp_bytes_to_id, self.rank))
        return out

    def encode_iterable(self, iterable) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i in self.vocab).decode("utf-8", errors="replace")
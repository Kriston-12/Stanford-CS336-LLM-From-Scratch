#include <unordered_map>
#include <vector>
#include <queue>
#include <memory>
#include <string>
#include <thread>

void add_vocab(std::unordered_map<int, std::string>& m, int k, const std::string& v) {
    m[k] = v;
}

void add_bytes_to_id(std::unordered_map<std::string, int>& m, const std::string& k, int v) {
    m[k] = v;
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
) {
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
    num_threads = std::min(n, num_threads);
    int chunk_size = (n + num_threads - 1) / num_threads;
    
    std::vector<std::thread> thread_pool;
    thread_pool.reserve(num_threads);
    
    std::vector<std::vector<int>> results(n); 

    for (int i = 0; i < num_threads; i++) {
        int start = i * chunk_size; 
        int end = std::min(start + chunk_size, n);
        if (start >= n) break;

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
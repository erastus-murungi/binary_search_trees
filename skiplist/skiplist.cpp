#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <map>
#include <chrono>


void test_height_gen();

static inline std::vector<int> *random_integers(int low, int high, uint size);

template<typename T>
struct skipnode {
    T value;
    std::vector<skipnode<T> *> next;

    explicit skipnode(T value) : value(value) {}

    skipnode(T value, uint8_t height) : value(value), next(height, nullptr) {}

    ~skipnode() {};

};

template<typename T>
class skiplist {
    uint max_height;
    size_t count;
    uint height;
    skipnode<T> header;


public:
    explicit skiplist(uint H = 32) : max_height(H), count(0), height(1), header(INT_MAX, max_height) {}

    skipnode<T> *access(T value);

    void insert(T value);

    bool find(T value);

    size_t length();

    uint genheight();

    void destroy();

    bool remove(T value);

    skipnode<T> *access_predecessor(T value);

    skipnode<T> *access_successor(T value);

    std::vector<T> *to_vector();

    ~skiplist() {destroy();};

};

template<typename T>
skipnode<T> *skiplist<T>::access(T value) {
        skipnode<T> *s = &header;
        int level;
        for (level = this->height - 1; level >= 0; level--) {
                while (s->next[0] && s->next[0].value <= value) {
                        s = s->next[level];
                }
        }
        return s;
}

template<typename T>
bool skiplist<T>::find(T value) {
        return access(value) != this->header;
}

template<typename T>
uint skiplist<T>::genheight() {
        std::random_device rd;
        std::mt19937 gen(rd());

        int c = 1;
        while ((gen() & (uint) 1) == 0 && c < this->max_height)
                c++;
        return c;
}

template<typename T>
size_t skiplist<T>::length() {
        return this->count;
}

template <typename T>
void skiplist<T>::destroy() {
        skipnode<T> *s, *next;
        s = header.next[0];
        while(s) {
                next = s->next[0];
                delete s;
                s = next;
        }

}

template<typename T>
void skiplist<T>::insert(T value) {

        uint h = genheight();
        auto *elt = new skipnode<T>(value, h);
        height = h > height ? h : height;

        int level;
        skipnode<T> *s = &header;

        for (level = height - 1; level >= h; level--) {
                while (s->next[level] && s->next[level]->value < value)
                        s = s->next[level];
        }
        for (; level >= 0; level--) {
                while (s->next[level] && s->next[level]->value < value)
                        s = s->next[level];

                elt->next[level] = s->next[level];
                s->next[level] = elt;
        }
        count += 1;
}


template <typename T>
bool skiplist<T>::remove(T value) {
        int level;
        skipnode<T> *s, *target;
        s = target = &header;

        for (level = height - 1; level >= 0; level--) {
                while (target->next[level] && target->next[level]->value < value)
                        target = target->next[level];
        }

        target = target->next[0];
        if (!target || target->value != value)
                return false;

        for (level = height - 1; level >= 0; level--) {
                while (s->next[level] && s->next[level]->value < value)
                        s = s->next[level];

                if (s->next[level] == target)
                        s->next[level] == target->next[level];
        }
        return true;
}

template<typename T>
skipnode<T> *skiplist<T>::access_predecessor(T value) {
        skipnode<T> *s = &header;
        int level;
        for (level = height - 1; level >= 0; level--)
                while (s->next[level] and s->next[level] < value)
                        s = s->next[level];
        return s;
}

template<typename T>
skipnode<T> *skiplist<T>::access_successor(T value) {
        skipnode<T> *s = access_predecessor(value);
        while (s->next and s->next->value <= value)
                s = s->next[0];
        return s;
}

template<typename T>
std::vector<T> *skiplist<T>::to_vector() {
        skipnode<T> *s = header.next[0];
        auto nums = new std::vector<T>();
        while (s) {
                nums->push_back(s->value);
                s = s->next[0];
        }
        return nums;
}


template<typename T>
std::ostream &operator<<(std::ostream &stream, skipnode<T> &node) {
        stream << node.value;
        return stream;
}


template<typename T>
std::ostream &operator<<(std::ostream &stream, skiplist<T> &st) {
        stream << "[";
        skipnode<T> s = st.header._nxt[0];
        while (s != st.header)
                stream << s.value << ", ";
        stream << "] \n";
        return stream;
}


template<typename K, typename V>
void print_map(std::map<K, V> const &m) {
        std::cout << "{ ";
        for (auto it = m.cbegin(); it != m.cend(); ++it) {
                std::cout << (*it).first << ": " << (*it).second << ", ";
        }
        std::cout << "}\n";
}

template<typename T>
void print_vector(std::vector<T> const &m) {
        std::cout << "{ ";
        for (auto it = m.cbegin(); it != m.cend(); ++it) {
                std::cout << *it << ", ";
        }
        std::cout << "}\n";
}



int main() {
        using namespace std;
        skipnode<int> s1(10);
        skiplist<int> st(10);

        auto start = chrono::steady_clock::now();
        vector<int> *nums = random_integers(0, 100, 10);
        auto end = chrono::steady_clock::now();
        cout << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;

        print_vector(*nums);
        for (int num: *nums){
                st.insert(num);
        }
        std::vector<int> *sorted = st.to_vector();
        print_vector(*sorted);
}

void test_height_gen() {
        skiplist<int> st(32);
        std::map<int, int> counter = std::map<int, int>();
        int h, i;
        for (i = 0; i < 10000; i++) {
                h = st.genheight();
                if (counter.find(h) != counter.end())
                        counter[h]++;
                else {
                        counter[h] = 1;
                }
        }
        print_map<int, int>(counter);
}

static inline std::vector<int>* random_integers(int low, int high, uint size) {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<int> dist(low, high);
        auto *numbers = new std::vector<int>();
        numbers->reserve(size);
        for (uint i = 0; i < size; i++) {
                numbers->emplace(numbers->begin() + i, dist(engine));
        }
        return numbers;
}

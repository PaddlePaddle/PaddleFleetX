#pragma once
#include <glog/logging.h>

namespace paddle {
namespace ps {

// Fast allocation and deallocation of objects by allocating them in chunks.
template<class T>
class ChunkAllocator {
public:
    explicit ChunkAllocator(size_t chunk_size = 64) {
        CHECK(sizeof(Node) == std::max(sizeof(void*), sizeof(T)));
        _chunk_size = chunk_size;
        _chunks = NULL;
        _free_nodes = NULL;
        _counter = 0;
    }
    ChunkAllocator(const ChunkAllocator&) = delete;
    ~ChunkAllocator() {
        while (_chunks != NULL) {
            Chunk* x = _chunks;
            _chunks = _chunks->next;
            free(x);
        }
    }
    template<class... ARGS>
    T* acquire(ARGS && ... args) {
        if (_free_nodes == NULL) {
            create_new_chunk();
        }

        T* x = (T*)(void*)_free_nodes;
        _free_nodes = _free_nodes->next;
        new(x) T(std::forward<ARGS>(args)...);
        _counter++;
        return x;
    }
    void release(T* x) {
        x->~T();
        Node* node = (Node*)(void*)x;
        node->next = _free_nodes;
        _free_nodes = node;
        _counter--;
    }
    size_t size() const {
        return _counter;
    }
private:
    struct alignas(T) Node {
        union {
            Node* next;
            char data[sizeof(T)];
        };
    };
    struct Chunk {
        Chunk* next;
        Node nodes[];
    };

    size_t _chunk_size; // how many elements in one chunk
    Chunk* _chunks; // a list
    Node* _free_nodes; // a list
    size_t _counter; // how many elements are acquired

    void create_new_chunk() {
        Chunk* chunk;
        posix_memalign((void**)&chunk, std::max<size_t>(sizeof(void*), alignof(Chunk)),
                sizeof(Chunk) + sizeof(Node) * _chunk_size);
        chunk->next = _chunks;
        _chunks = chunk;

        for (size_t i = 0; i < _chunk_size; i++) {
            Node* node = &chunk->nodes[i];
            node->next = _free_nodes;
            _free_nodes = node;
        }
    }
};

} //namespace ps
} //namespace paddle

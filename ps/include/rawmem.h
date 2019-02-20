#pragma once
#include <cstdlib>
#include <memory>

namespace paddle {
    namespace ps {
        class RawMemory {
        public:
            explicit RawMemory(size_t size) : _data_ptr(NULL), _size(size) {};
            ~RawMemory() {};
            virtual const void * data() { return _data_ptr; }
        protected:
            void * _data_ptr;
            size_t _size;
         };
     }
}

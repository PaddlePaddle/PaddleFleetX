#pragma once
#include "rawmem.h"
#include <memory>
#include <immintrin.h>
#include <vector>

namespace paddle {
    namespace ps {
        template <typename Dtype>
        class Blob {
        public:
            Blob() : _data(), _count(0), _capacity(-1) {}
            virtual ~Blob() {}
            explicit Blob(const std::vector<int>& shape);
            virtual void reshape(const std::vector<int>& shape);
            inline const std::vector<int>& shape() const { return _shape; }
            const Dtype* data() const;
            Dtype* mutable_data();
        protected:
            std::shared_ptr<RawMemory> _data;
            int _count;
            int _capacity;
            std::vector<int> _shape;
        };
    }
}

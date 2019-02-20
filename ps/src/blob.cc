#include "blob.h"
#include "rawmem.h"

namespace paddle {
namespace ps {
    template <typename Dtype>
    void Blob<Dtype>::reshape(const std::vector<int>& shape) {
        _count = 1;
        _shape.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            _count *= shape[i];
            _shape[i] = shape[i];
        }
        if (_count > _capacity) {
            _capacity = _count;
            _data.reset(new RawMemory(_capacity * sizeof(Dtype)));
        }
    }
    
    template <typename Dtype>
    const Dtype* Blob<Dtype>::data() const {
        return (const Dtype*)_data->data();
    }
    
    template class Blob<float>;
    template class Blob<int>;
    template class Blob<unsigned long>;
}
}

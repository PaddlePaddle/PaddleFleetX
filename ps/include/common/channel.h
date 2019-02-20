#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_CHANNEL_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_CHANNEL_H 

#include <memory>
#include <mutex>
#include <vector>
#include <deque>
#include <condition_variable>
#include <glog/logging.h>
#include "common/expect.h"

namespace paddle {
namespace ps {

template<class T>
class ChannelObject {
public:
    ChannelObject() {
    }
    explicit ChannelObject(size_t capacity) { // capacity can be zero
        _capacity = std::min(max_capacity(), capacity);
    }
    size_t capacity() {
        return _capacity; // atomic
    }
    void set_capacity(size_t x) { // capacity can be zero
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = std::min(max_capacity(), x);
        notify();
    }
    size_t block_size() {
        return _block_size; // atomic
    }
    void set_block_size(size_t x) {
        CHECK(x >= 1);
        std::lock_guard<std::mutex> lock(_mutex);
        _block_size = x;
    }
    template<class TT>
    void inherit_from(const std::shared_ptr<ChannelObject<TT>>& other) {
        std::lock_guard<std::mutex> lock(_mutex);
        _capacity = other->capacity();
        _block_size = other->block_size();
    }
    bool closed() {
        return _closed; // atomic
    }
    void open() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = false;
        notify();
    }
    void close() {
        std::lock_guard<std::mutex> lock(_mutex);
        _closed = true;
        notify();
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _data.size();
    }
    bool empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return empty_unlocked();
    }
    // blocking operation
    bool get(T& val) {
        return read(1, &val) != 0;
    }
    // blocking operation
    // returns 0 if the channel is closed and empty
    size_t read(size_t n, T* p) {
        if (n == 0) {
            return 0;
        }

        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = read(n, p, lock);
        notify();
        return finished;
    }
    // blocking operation
    bool put(T && val) {
        return write_move(1, &val) != 0;
    }
    // blocking operation
    bool put(const T& val) {
        return write(1, &val) != 0;
    }
    // blocking operation
    // returns value less than n if the channel is closed
    size_t write(size_t n, const T* p) {
        if (n == 0) {
            return 0;
        }

        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = write(n, p, lock);
        notify();
        return finished;
    }
    // write_move() will clear original contents of p
    size_t write_move(size_t n, T* p) {
        if (n == 0) {
            return 0;
        }

        std::unique_lock<std::mutex> lock(_mutex);
        size_t finished = write_move(n, p, lock);
        notify();
        return finished;
    }
    size_t read(std::vector<T>& p) {
        p.resize(_block_size);
        size_t finished = read(p.size(), &p[0]);
        p.resize(finished);
        return finished;
    }
    size_t read_all(std::vector<T>& p) {
        p.clear();
        size_t finished = 0;
        size_t n = 0;

        do {
            n = _block_size; // _block_size may change anytime
            p.resize(finished + n);
            n = read(n, &p[finished]);
            finished += n;
        } while (n != 0);

        p.resize(finished);
        return finished;
    }
    size_t write(const std::vector<T>& p) {
        return write(p.size(), &p[0]);
    }
    size_t write(std::vector<T> && p) {
        return write_move(p.size(), &p[0]);
    }
private:
    size_t _capacity = max_capacity();
    size_t _block_size = 1024;
    bool _closed = false;

    std::mutex _mutex;
    std::deque<T> _data;
    size_t _reading_count = 0;
    int _empty_waiters = 0;
    int _full_waiters = 0;
    std::condition_variable _empty_cond;
    std::condition_variable _full_cond;

    static constexpr size_t max_capacity() {
        return std::numeric_limits<size_t>::max() / 2;
    }
    void notify() {
        if (_empty_waiters != 0 && (!empty_unlocked() || _closed)) {
            _empty_cond.notify_one();
        }

        if (_full_waiters != 0 && (!full_unlocked() || _closed)) {
            _full_cond.notify_one();
        }
    }
    bool empty_unlocked() {
        return _data.empty();
    }
    bool full_unlocked() {
        return _data.size() >= _capacity + _reading_count;
    }
    bool wait_for_read(std::unique_lock<std::mutex>& lock) {
        while (unlikely(empty_unlocked() && !_closed)) {
            if (_full_waiters != 0) {
                _full_cond.notify_one();
            }

            _empty_waiters++;
            _empty_cond.wait(lock);
            _empty_waiters--;
        }

        return !empty_unlocked();
    }
    bool wait_for_write(std::unique_lock<std::mutex>& lock) {
        while (unlikely(full_unlocked() && !_closed)) {
            if (_empty_waiters != 0) {
                _empty_cond.notify_one();
            }

            _full_waiters++;
            _full_cond.wait(lock);
            _full_waiters--;
        }

        return !_closed;
    }
    size_t read(size_t n, T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;
        CHECK(n <= max_capacity() - _reading_count);
        _reading_count += n;

        while (finished < n && wait_for_read(lock)) {
            size_t m = std::min(n - finished, _data.size());

            for (size_t i = 0; i < m; i++) {
                p[finished++] = std::move(_data.front());
                _data.pop_front();
            }

            _reading_count -= m;
        }

        _reading_count -= n - finished;
        return finished;
    }
    size_t write(size_t n, const T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;

        while (finished < n && wait_for_write(lock)) {
            size_t m = std::min(n - finished, _capacity + _reading_count - _data.size());

            for (size_t i = 0; i < m; i++) {
                _data.push_back(p[finished++]);
            }
        }

        return finished;
    }
    size_t write_move(size_t n, T* p, std::unique_lock<std::mutex>& lock) {
        size_t finished = 0;

        while (finished < n && wait_for_write(lock)) {
            size_t m = std::min(n - finished, _capacity + _reading_count - _data.size());

            for (size_t i = 0; i < m; i++) {
                _data.push_back(std::move(p[finished++]));
            }
        }

        return finished;
    }
};

template<class T>
using Channel = std::shared_ptr<ChannelObject<T>>;

template<class T>
Channel<T> make_channel(size_t capacity = std::numeric_limits<size_t>::max()) {
    return std::make_shared<ChannelObject<T>>(capacity);
}

template<class T, class TT>
Channel<T> make_channel(const Channel<TT>& other) {
    CHECK(other);
    Channel<T> chan = std::make_shared<ChannelObject<T>>();
    chan->inherit_from(other);
    return chan;
}

// @NOTE ChannelReader is a wrapper for quick read channel with a buffer. It will read a block
// data from channel, but user can get data one by one. So it is important to notice that user
// must call operator>> until false, or call get_buffer_remain until false to make sure the
// buffered data all readed.
template<class T>
class ChannelReader {
public:
    explicit ChannelReader(const Channel<T>& channel = nullptr) {
        reset(channel);
    }
    ~ChannelReader() {
        CHECK(_cursor == 0) << "Forgot to read buffer data";
    }
    const Channel<T>& channel() {
        return _channel;
    }
    void reset(const Channel<T>& channel) {
        _channel = channel;
        _cursor = 0;
        _failed = !channel;
    }
    // whether there were read failed
    operator bool() {
        return !_failed;
    }
    ChannelReader<T>& operator>>(T& val) {
        if (_failed) {
            return *this;
        }

        if (_cursor >= _buffer.size()) {
            _cursor = 0;

            if (_channel->read(_buffer) == 0) {
                _failed = true;
                return *this;
            }
        }

        val = std::move(_buffer[_cursor++]);
        return *this;
    }
    bool get_buffer_remain(T& val) {
        if (_cursor >= _buffer.size()) {
            _cursor = 0;
            return false;
        }
        val = std::move(_buffer[_cursor++]);
        return true;
    }
private:
    Channel<T> _channel;
    std::vector<T> _buffer;
    size_t _cursor = 0;
    bool _failed = true;
};

template<class T>
class ChannelWriter {
public:
    explicit ChannelWriter(const Channel<T>& channel = nullptr) {
        reset(channel);
    }
    ~ChannelWriter() {
        CHECK(_buffer.empty()) << "Forgot to flush";
    }
    const Channel<T>& channel() {
        return _channel;
    }
    void reset(const Channel<T>& channel) {
        CHECK(_buffer.empty()) << "Forgot to flush";
        _channel = channel;
        _buffer.clear();
        _failed = !channel;
    }
    // whether there were write failed
    operator bool() {
        return !_failed;
    }
    ChannelWriter<T>& operator<<(T && val) {
        if (_failed) {
            return *this;
        }

        _buffer.push_back(std::move(val));

        if (_buffer.size() >= _channel->block_size()) {
            flush();
        }

        return *this;
    }
    ChannelWriter<T>& operator<<(const T& val) {
        if (_failed) {
            return *this;
        }

        _buffer.push_back(val);

        if (_buffer.size() >= _channel->block_size()) {
            flush();
        }

        return *this;
    }
    void flush() {
        if (_failed || _buffer.empty()) {
            _buffer.clear();
            return;
        }

        _failed |= _channel->write_move(_buffer.size(), &_buffer[0]) != _buffer.size();
        _buffer.clear();
    }
private:
    Channel<T> _channel;
    std::vector<T> _buffer;
    bool _failed = true;
};

// only used for range-for loop
// for (auto& x : chan) {...}
template<class T>
struct ChannelIterator {
    std::shared_ptr<ChannelReader<T>> reader;
    T data;
    void operator++() {
        CHECK(reader);

        if (!(*reader >> data)) {
            reader = nullptr;
        }
    }
    T& operator*() {
        return data;
    }
    friend bool operator==(const ChannelIterator<T>& a, const ChannelIterator<T>& b) {
        return a.reader == b.reader;
    }
    friend bool operator!=(const ChannelIterator<T>& a, const ChannelIterator<T>& b) {
        return a.reader != b.reader;
    }
};

template<class T>
ChannelIterator<T> begin(const Channel<T>& chan) {
    ChannelIterator<T> it {std::make_shared<ChannelReader<T>>(chan), T()};
    ++it;
    return it;
}

template<class T>
ChannelIterator<T> end(const Channel<T>& chan) {
    return {nullptr, T()};
}

}
}
#endif //

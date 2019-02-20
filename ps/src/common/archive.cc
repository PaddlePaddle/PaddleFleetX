#include "common/archive.h"
//#include "boost/spirit/include/qi.hpp"
//#include "boost/lexical_cast.hpp"

namespace paddle {
namespace ps {

ArchiveBase::ArchiveBase(ArchiveBase && other) :
        _buffer(other._buffer),
        _cursor(other._cursor),
        _finish(other._finish),
        _limit(other._limit),
        _deleter(std::move(other._deleter)) {
    other._buffer = NULL;
    other._cursor = NULL;
    other._finish = NULL;
    other._limit = NULL;
    other._deleter = nullptr;
}

ArchiveBase& ArchiveBase::operator=(ArchiveBase && other) {
    if (this != &other) {
        free_buffer();
        _buffer = other._buffer;
        _cursor = other._cursor;
        _finish = other._finish;
        _limit = other._limit;
        _deleter = std::move(other._deleter);
        other._buffer = NULL;
        other._cursor = NULL;
        other._finish = NULL;
        other._limit = NULL;
        other._deleter = nullptr;
    }
    return *this;
}

void ArchiveBase::set_buffer(char* buffer, size_t length, size_t capacity,
        std::function<void(char*)> && deleter) {
    CHECK(length <= capacity);
    free_buffer();
    _buffer = buffer;
    _cursor = _buffer;
    _finish = buffer + length;
    _limit = buffer + capacity;
    _deleter = std::move(deleter);
}

void ArchiveBase::reserve(size_t newcap) {
    if (newcap > capacity()) {
        char* newbuf = NULL;
        newbuf = new char[newcap];
        CHECK(newbuf != nullptr) << "out of memory"; 
        if (length() > 0) {
            memcpy(newbuf, _buffer, length());
        }

        _cursor = newbuf + (_cursor - _buffer);
        _finish = newbuf + (_finish - _buffer);
        _limit = newbuf + newcap;
        free_buffer();
        _buffer = newbuf;
        _deleter = std::default_delete<char[]>();
    }
}

// put template function in cpp is a hack for avoid include boost head files in head files
// which will make compile very slow.
//template<class T>
//void Archive<TextArchiveType>::get_arithmetic(T& x) {
//    CHECK(boost::spirit::qi::phrase_parse(_cursor, _finish, boost::spirit::auto_,
//            boost::spirit::ascii::char_(' '), x));
//    CHECK(_cursor == _finish || *_cursor == '\t');
//
//    if (_cursor < _finish) {
//        ++_cursor;
//    }
//}
//template void Archive<TextArchiveType>::get_arithmetic<int16_t>(int16_t&);
//template void Archive<TextArchiveType>::get_arithmetic<uint16_t>(uint16_t&);
//template void Archive<TextArchiveType>::get_arithmetic<int32_t>(int32_t&);
//template void Archive<TextArchiveType>::get_arithmetic<uint32_t>(uint32_t&);
//template void Archive<TextArchiveType>::get_arithmetic<int64_t>(int64_t&);
//template void Archive<TextArchiveType>::get_arithmetic<uint64_t>(uint64_t&);
//template void Archive<TextArchiveType>::get_arithmetic<float>(float&);
//template void Archive<TextArchiveType>::get_arithmetic<double>(double&);
//template void Archive<TextArchiveType>::get_arithmetic<signed char>(signed char&);
//template void Archive<TextArchiveType>::get_arithmetic<unsigned char>(unsigned char&);

/*
template<class T>
void Archive<TextArchiveType>::get_arithmetic_space(T& x) { // deprecated
    CHECK(boost::spirit::qi::phrase_parse(_cursor, _finish, boost::spirit::auto_,
            boost::spirit::ascii::space, x));
    CHECK(_cursor == _finish || isspace(*(_cursor - 1)));
}
template void Archive<TextArchiveType>::get_arithmetic_space<int16_t>(int16_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<uint16_t>(uint16_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<int32_t>(int32_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<uint32_t>(uint32_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<int64_t>(int64_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<uint64_t>(uint64_t&);
template void Archive<TextArchiveType>::get_arithmetic_space<float>(float&);
template void Archive<TextArchiveType>::get_arithmetic_space<double>(double&);
template void Archive<TextArchiveType>::get_arithmetic_space<signed char>(signed char&);
template void Archive<TextArchiveType>::get_arithmetic_space<unsigned char>(unsigned char&);

template<class T>
void Archive<TextArchiveType>::put_arithmetic(const T& x) {
    // boost::spirit has bug in generating string from floating point number
    // (could crash for some denormalized value)
    typedef std::array<char, 64> str_t;
    prepare_write(sizeof(str_t));
    *(str_t*)finish() = boost::lexical_cast<str_t>(x);
    char* i = finish() + strlen(finish());
    *i = '\t';
    set_finish(i + 1);
}
template void Archive<TextArchiveType>::put_arithmetic<int16_t>(const int16_t&);
template void Archive<TextArchiveType>::put_arithmetic<uint16_t>(const uint16_t&);
template void Archive<TextArchiveType>::put_arithmetic<int32_t>(const int32_t&);
template void Archive<TextArchiveType>::put_arithmetic<uint32_t>(const uint32_t&);
template void Archive<TextArchiveType>::put_arithmetic<int64_t>(const int64_t&);
template void Archive<TextArchiveType>::put_arithmetic<uint64_t>(const uint64_t&);
template void Archive<TextArchiveType>::put_arithmetic<float>(const float&);
template void Archive<TextArchiveType>::put_arithmetic<double>(const double&);
template void Archive<TextArchiveType>::put_arithmetic<signed char>(const signed char&);
template void Archive<TextArchiveType>::put_arithmetic<unsigned char>(const unsigned char&);
*/

}
} // namespace 

// This file is part of Miscellaneous Container Templates.
//
//             https://launchpad.net/libmct/
//
// Copyright (c) 2009, 2010, 2011, 2012, 2013 Paul Pogonyshev.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef MCT_IMPL_UTILS_HPP
#define MCT_IMPL_UTILS_HPP


#include <cmath>
#include <cstddef>
#include <iterator>

#if MCT_HAVE_TYPE_TRAITS
# include MCT_TYPE_TRAITS_HEADER
#endif

#if MCT_ENABLE_DEBUGGING || MCT_CHECK_PRECONDITIONS || MCT_SELF_VALIDATION
# define MCT_DEBUGGING_MEMBERS  1
#else
# define MCT_DEBUGGING_MEMBERS  0
#endif


#if MCT_CHECK_PRECONDITIONS

# define MCT_PRECONDITION(condition, error_message)     \
  do                                                    \
    {                                                   \
      if (!(condition))                                 \
        throw std::logic_error (error_message);         \
    }                                                   \
  while (false)

#else
# define MCT_PRECONDITION(condition, error_message)
#endif


#if MCT_SELF_VALIDATION
# define MCT_VALIDATION(statement)      (statement)
#else
# define MCT_VALIDATION(statement)
#endif


#if defined (__GNUC__)
# define MCT_DEPRECATED                 __attribute__((deprecated))
#else
# define MCT_DEPRECATED
#endif


#define MCT_UNUSED(x)                   static_cast <void> (x)


namespace mct
{

  namespace impl
  {

# if MCT_HAVE_TYPE_TRAITS

    using MCT_TYPE_TRAITS_NAMESPACE::false_type;
    using MCT_TYPE_TRAITS_NAMESPACE::true_type;

    using MCT_TYPE_TRAITS_NAMESPACE::is_same;
    using MCT_TYPE_TRAITS_NAMESPACE::is_integral;

# else  // !MCT_HAVE_TYPE_TRAITS

    struct false_type
    {
      static  const bool  value = false;
    };

    struct true_type
    {
      static  const bool  value = true;
    };


    template <typename type1, typename type2>
    struct is_same : false_type
    { };

    template <typename type>
    struct is_same <type, type> : true_type
    { };


    template <typename type>
    struct is_integral : false_type
    { };

    template <>
    struct is_integral <bool> : true_type
    { };

    template <>
    struct is_integral <char> : true_type
    { };
    template <>
    struct is_integral <signed char> : true_type
    { };
    template <>
    struct is_integral <unsigned char> : true_type
    { };

    template <>
    struct is_integral <signed short> : true_type
    { };
    template <>
    struct is_integral <unsigned short> : true_type
    { };

    template <>
    struct is_integral <signed int> : true_type
    { };
    template <>
    struct is_integral <unsigned int> : true_type
    { };

    template <>
    struct is_integral <signed long> : true_type
    { };
    template <>
    struct is_integral <unsigned long> : true_type
    { };

#   if MCT_HAVE_LONG_LONG
    template <>
    struct is_integral <signed long long> : true_type
    { };
    template <>
    struct is_integral <unsigned long long> : true_type
    { };
#   endif


    // FIXME: Improve if needed.  Only used if the standard library doesn't provide type
    //        traits.
    template <typename type>
    struct has_trivial_destructor : is_integral <type>
    { };

# endif  // !MCT_HAVE_TYPE_TRAITS


    template <typename type>
    inline  type
    round_up_to_power_of_2 (type number)
    {
      if (number & (number - 1))
        {
          for (number <<= 1; number & (number - 1);)
            number &= (number - 1);
        }

      return number;
    }

    template <typename type>
    inline  type
    round_down_to_power_of_2 (type number)
    {
      while (number & (number - 1))
        number &= (number - 1);

      return number;
    }


    template <typename Iterator>
    inline  typename std::iterator_traits <Iterator>::difference_type
    forward_distance (Iterator /* first */, Iterator /* last */, std::input_iterator_tag)
    {
      // Cannot determine in general case.
      return 0;
    }

    template <typename Iterator>
    inline  typename std::iterator_traits <Iterator>::difference_type
    forward_distance (Iterator first, Iterator last, std::forward_iterator_tag)
    {
      return std::distance (first, last);
    }

    template <typename Iterator>
    inline  typename std::iterator_traits <Iterator>::difference_type
    forward_distance (Iterator first, Iterator last)
    {
      typedef  typename std::iterator_traits <Iterator>::iterator_category  category;
      return forward_distance (first, last, category ());
    }

    template <typename Iterator>
    inline  typename std::iterator_traits <Iterator>::difference_type
    forward_distance_scaled (Iterator first, Iterator last, float inverted_scale)
    {
      typedef  typename std::iterator_traits <Iterator>::iterator_category  category;
      return (static_cast <typename std::iterator_traits <Iterator>::difference_type>
              (std::ceil (forward_distance (first, last, category ()) / inverted_scale)));
    }


    template <bool condition, typename Type = void>
    struct enable_if
    { };

    template <typename Type>
    struct enable_if <true, Type>
    {
      typedef  Type  type;
    };


    template <int size_t_size = sizeof (std::size_t)>
    struct size_t_constants
    {
      static  const std::size_t  HUGE_PRIME = static_cast <std::size_t> (0x1a4548ec34f6e06fUL);
    };

    template <>
    struct size_t_constants <4>
    {
      static  const std::size_t  HUGE_PRIME = static_cast <std::size_t> (0x3874e607UL);
    };

  }

}


#if MCT_HAVE_TYPE_TRAITS

// While you can't legally insert names into 'std' namespace, it should be fine in this
// particular case.  This seems to be the only way to handle the trait renaming between
// C++0x drafts and the final C++11 standard without configuration.  Unfortunately, we
// cannot "inject" names into MCT_TYPE_TRAITS_NAMESPACE here, because the latter can
// contain '::' ('std::tr1'), so we always do this to 'std'.  The configuration is
// partially ignored in this case, which seems to be the lesser of evil options we have.
namespace std
{
  template <typename>
  struct has_trivial_destructor;

  template <typename>
  struct is_trivially_destructible;
}


namespace mct
{

  namespace impl
  {

#   if defined (__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 3) || __GNUC__ > 4)

    // On GCC TR1 implementation provides subpar 'has_trivial_destructor', so we use the
    // built-in instead.
    template <typename type>
    struct has_trivial_destructor
    {
      static  const bool  value = __has_trivial_destructor (type);
    };

#   else

    template <typename type, typename>
    struct htd_helper : false_type
    { };

    template <typename type>
    struct htd_helper <type, typename enable_if <std::is_trivially_destructible <type>::value, void>::type>
      : true_type
    { };

    template <typename type>
    struct htd_helper <type, typename enable_if <std::has_trivial_destructor <type>::type::value, char>::type>
      : true_type
    { };

    template <typename type>
    struct htd_helper <type, typename enable_if <MCT_TYPE_TRAITS_NAMESPACE
                                                 ::has_trivial_destructor <type>::type::value, int>::type>
      : true_type
    { };


    using MCT_TYPE_TRAITS_NAMESPACE::integral_constant;

    // We define our internal "compatibility" trait as true if any of
    // 'std::is_trivially_destructible' (C++11), 'std::has_trivial_destructor' (previous
    // C++0x drafts) or the corresponding trait in the configured namespace (may or may
    // not be 'std') is true.
    template <typename type>
    struct has_trivial_destructor : integral_constant <bool, (htd_helper    <type, void>::value
                                                              || htd_helper <type, char>::value
                                                              || htd_helper <type, int> ::value)>
    { };

#   endif

  }

}

#endif  // MCT_HAVE_TYPE_TRAITS


#endif  // Multi-inclusion guard.


// Local variables:
// mode: c++
// c-basic-offset: 2
// indent-tabs-mode: nil
// fill-column: 90
// End:

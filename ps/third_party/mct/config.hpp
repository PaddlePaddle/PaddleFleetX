#ifndef MCT_CONFIG_HPP
#define MCT_CONFIG_HPP

// MCT version.
#define MCT_MAJOR_VERSION               1
#define MCT_MINOR_VERSION               6
#define MCT_PATCH_VERSION               2
#define MCT_VERSION_STRING              "1.6.2"

// Library support, including stdlib.
#if !defined (MCT_HASH_HEADER) || !defined (MCT_HASH_NAMESPACE)
#  define MCT_HASH_HEADER               <functional>
#  define MCT_HASH_NAMESPACE            std
//#  define MCT_HASH_HEADER               <boost/functional/hash.hpp>
//#  define MCT_HASH_NAMESPACE            boost
#endif

#if !defined (MCT_TYPE_TRAITS_HEADER) || !defined (MCT_TYPE_TRAITS_NAMESPACE)
#  define MCT_HAVE_TYPE_TRAITS          1
#  define MCT_TYPE_TRAITS_HEADER        <type_traits>
#  define MCT_TYPE_TRAITS_NAMESPACE     std
//#  define MCT_TYPE_TRAITS_HEADER        <boost/type_traits.hpp>
//#  define MCT_TYPE_TRAITS_NAMESPACE     boost
#else
#  define MCT_HAVE_TYPE_TRAITS          1
#endif

// Compiler capabilities.
#if !defined (MCT_CXX0X_SUPPORTED)
#  define MCT_CXX0X_SUPPORTED           1
#endif

#if !defined (MCT_HAVE_LONG_LONG)
#  define MCT_HAVE_LONG_LONG            1
#endif

#endif  // Multi-inclusion guard.


// Local variables:
// mode: c++
// c-basic-offset: 2
// indent-tabs-mode: nil
// fill-column: 90
// End:

/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Utilities and statistics for the treeio::Tree class.
 */

#ifndef TREE_UTILS_H
#define TREE_UTILS_H

#include <array>
#include <algorithm>
#include <execution>
#include <vector>
#include <stack>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <deque>
#include <exception>
#include <functional>
#include <set>
#include <map>
#include <memory>
#include <mutex>
#include <cinttypes>
#include <filesystem>
#include <numeric>
#include <utility>
#include <cmath>
#include <ctime>
#include <random>

#include <glm/glm.hpp>
#define HAS_GLM

#include "TreeIO/TreeIO.h"

#include "TreeMath.h"
#include "TreeTemplateUtils.h"

/// @brief Helper for signalling to the compiler that we know we didn't use a thing...
#define TREE_UNUSED(x) (void)(x)

namespace treeutil 
{

/// @brief Helper class which allows checkpointing for given type.
template <typename T>
class VersionCheckpoint
{
public:
    /// Type used for identifying different versions.
    using StampT = std::size_t;
    /// Type used for identifying different objects.
    using IdentifierT = const void*;
    /// Reference to this whole type.
    using ThisT = VersionCheckpoint<T>;

    /// @brief Initialize checkpoint different to all other checkpoints.
    VersionCheckpoint() = default;

    /// @brief Target checkpointed object is not destroyed.
    ~VersionCheckpoint() = default;
        
    /// @brief Initialize checkpoint for given object.
    VersionCheckpoint(const T *obj, const StampT &v = { }) : 
        mStamp{ v }, mIdentifier{ obj } { }

    /// @brief Initialize checkpoint for given object.
    VersionCheckpoint(const T &obj, const StampT &v = { }) : 
        VersionCheckpoint(&obj, v) { }

    /// @brief Initialize checkpoint for given object.
    VersionCheckpoint(const WrapperPtrT<const T> &obj, const StampT &v = { }) : 
        VersionCheckpoint(obj.get(), v) { }
    
    // Copy and copy assignment: 
    VersionCheckpoint(const ThisT &other) : 
        VersionCheckpoint() { *this = other; }
    ThisT &operator=(const ThisT &other)
    { mStamp = other.mStamp; mIdentifier = other.mIdentifier; return *this; }
    
    /**
     * @brief Check whether the checked checkpoint is older or bound to other 
     * object than the against checkpoint.
     */
    static bool isCheckpointOlderOrInvalid(const ThisT &check, const ThisT &against)
    { return check.mIdentifier != against.mIdentifier || check.mStamp < against.mStamp; }

    /**
     * @brief Check whether the checked checkpoint is newer or bound to other 
     * object than the against checkpoint.
     */
    static bool isCheckpointNewerOrInvalid(const ThisT &check, const ThisT &against)
    { return check.mIdentifier != against.mIdentifier || check.mStamp > against.mStamp; }

    /// @brief Check if this checkpoint is older or invalid against given checkpoint.
    bool olderOrInvalid(const ThisT &against) const 
    { return isCheckpointOlderOrInvalid(*this, against); }

    /// @brief Check if this checkpoint is newer or invalid against given checkpoint.
    bool newerOrInvalid(const ThisT &against) const 
    { return isCheckpointNewerOrInvalid(*this, against); }

    /// @brief Check if this checkpoint is older or invalid against given checkpoint.
    bool operator<(const ThisT &other) const
    { return olderOrInvalid(other); }

    /// @brief Check if this checkpoint is newer or invalid against given checkpoint.
    bool operator>(const ThisT &other) const
    { return newerOrInvalid(other); }

    /// @brief Check if this checkpoint is older or equal to given checkpoint.
    bool operator<=(const ThisT &other) const
    { return olderOrInvalid(other) || operator==(other); }

    /// @brief Check if this checkpoint is newer or equal to given checkpoint.
    bool operator>=(const ThisT &other) const
    { return newerOrInvalid(other) || operator==(other); }

    /// @brief Check whether both checkpoints use the same object and stamp.
    bool operator==(const ThisT &other) const
    { return mIdentifier == other.mIdentifier && mStamp == other.mStamp; }

    /// @brief Check if the checkpoints use different object or stamp.
    bool operator!=(const ThisT &other) const
    { return !(*this == other); }

    /// @brief Pre-increment version of going to the next stamp.
    VersionCheckpoint &operator++()
    { mStamp++; return *this; }

    /// @brief post-increment version of going to the next stamp.
    VersionCheckpoint operator++(int)
    { VersionCheckpoint bck{ *this }; operator++(); return bck; }

    /// @brief Get current stamp value.
    StampT version() const
    { return mStamp; }

    /// @brief Get current identifier value.
    IdentifierT identifier() const
    { return mIdentifier; }
private:
    /// Value of a stamp considered to be null.
    static constexpr const auto NULL_STAMP{ StampT{ } };
    /// Value of an identifier considered to be null.
    static constexpr const auto NULL_IDENTIFIER{ IdentifierT{ } };

    /// Current stamp version.
    StampT mStamp{ NULL_STAMP };
    /// Currently referenced object.
    IdentifierT mIdentifier{ NULL_IDENTIFIER };
protected:
}; // class VersionCheckpoint

namespace impl
{

/// @brief Dummy type used for demarking block start for the serializer.
struct SerializerBlockStart
{ };
/// @brief Dummy type used for demarking block end for the serializer.
struct SerializerBlockEnd
{ };

/// Symbol representing start of a serialized block.
static constexpr char SERIALIZER_BLOCK_START{ '(' };
/// Symbol representing end of a serialized block.
static constexpr char SERIALIZER_BLOCK_END{ ')' };
/// Symbol used as divider between serialized values.
static constexpr char SERIALIZER_DIVIDER{ ';' };

} // namespace impl

/// @brief Simple helper for value serialization.
class Serializer
{
public:
    /// @brief Dummy type used for demarking block start for the serializer.
    using BlockStartT = impl::SerializerBlockStart;
    /// @brief Dummy value used for demarking block start for the serializer.
    static constexpr auto BlockStart{ impl::SerializerBlockStart{ } };
    /// @brief Dummy type used for demarking block end for the serializer.
    using BlockEndT = impl::SerializerBlockEnd;
    /// @brief Dummy value used for demarking block end for the serializer.
    static constexpr auto BlockEnd{ impl::SerializerBlockEnd{ } };

    Serializer() = default;
    ~Serializer() = default;

    /// @brief Reset current serialization to empty string.
    inline void reset();
    /// @brief Get current version of serialized string.
    inline std::string str() const;

    /// @brief Start writing a new block.
    inline void startBlock();
    /// @brief End current block.
    inline void endBlock();

    /// @brief Serialize given value by using its operator<<(std::ostream&).
    template <typename T>
    inline void serialize(const T &v);

    /// @brief Stream-like serialization. Uses serialize(v). Use BlockStart and BlockEnd for start/endBlock().
    template <typename T>
    inline Serializer &operator<<(const T &v);
private:
    /// Internal string stream used for the serialization.
    std::stringstream mSs{ };
    /// Flag used to determine whether a value has been written to string stream.
    bool mValueWritten{ false };
protected:
}; // class Serializer

/// @brief Simple helper for value deserialization.
class Deserializer
{
public:
    /// @brief Dummy type used for demarking block start for the serializer.
    using BlockStartT = impl::SerializerBlockStart;
    /// @brief Dummy value used for demarking block start for the serializer.
    static constexpr auto BlockStart{ impl::SerializerBlockStart{ } };
    /// @brief Dummy type used for demarking block end for the serializer.
    using BlockEndT = impl::SerializerBlockEnd;
    /// @brief Dummy value used for demarking block end for the serializer.
    static constexpr auto BlockEnd{ impl::SerializerBlockEnd{ } };
    /// @brief Exception thrown when some unexpected value is found.
    struct DeserializerException : public std::runtime_error
    {
        DeserializerException(const std::string &msg):
            std::runtime_error{ msg } { }
    }; // struct DeserializerException

    Deserializer() = default;
    ~Deserializer() = default;

    /// @brief Load serialized data.
    inline Deserializer(const std::string &serialized);

    /// @brief Load serialized data.
    inline void load(const std::string &serialized);

    /// @brief Start reading a block.
    inline void startBlock();
    /// @brief End reading current block.
    inline void endBlock();

    /// @brief Deserialize value of given type by using its operator>>(std::ostream&).
    template <typename T>
    inline T deserialize();

    /// @brief Stream-like deserialization. Uses deserialize(v). Use BlockStart and BlockEnd for start/endBlock().
    template <typename T>
    inline Deserializer &operator>>(T &v);
private:
    /// @brief Read a character from given stream.
    inline char readChar(std::stringstream &stream);
    /// @brief Check whether read character corresponds to the expectation and throw error if they are different.
    inline void checkCharThrow(char readValue, char expectedValue);

    /// Internal string stream used for the deserialization.
    std::stringstream mSs{ };
    /// Flag used to determine whether a value has been read from string stream.
    bool mValueRead{ false };
protected:
}; // class Deserializer

/// @brief Map which allows mapping of keys to any number of typed values. Each type has its own map.
template <typename KeyT, typename AttributeKeyT = KeyT>
class TemplateMap
{
public:
    /// Key type used for discerning elements within the map.
    using KT = KeyT;
    /// Key type used for discerning attributes associated with one element.
    using AKT = AttributeKeyT;

    /// @brief Create empty template map.
    TemplateMap();
    /// @brief Cleanup and destroy.
    ~TemplateMap();

    // Allow copying and moving.
    TemplateMap(const TemplateMap &other) = default;
    TemplateMap &operator=(const TemplateMap &other) = default;
    TemplateMap(TemplateMap &&other) = default;
    TemplateMap &operator=(TemplateMap &&other) = default;

    /// @brief Clear the map completely.
    void clear();

    /// @brief Does the map contain any elements?
    bool empty();

    /// @brief Does element with given key have associated value of provided type?
    template <typename T>
    bool has(const KT &element, const AKT &attribute) const;

    /// @brief Get value of given type for provided key. The value will be created if it does not exist.
    template <typename T>
    T &get(const KT &element, const AKT &attribute) const;
private:
    // Forward declaration.
    template <typename T>
    struct AttributeValue;

    /// @brief Base type for all value holders.
    struct AttributeValueBase
    {
        virtual ~AttributeValueBase() = default;
        AttributeValueBase(const AKT &name);

        /// @brief Cast this holder into specialized one.
        template <typename T>
        AttributeValue<T> *specialize();

        /// Name of the held attribute.
        AKT name{ };
    }; // struct AttributeValueBase

    /// @brief Holder of a concrete value.
    template <typename T>
    struct AttributeValue : public AttributeValueBase
    {
        virtual ~AttributeValue() = default;
        AttributeValue(const AKT &name);

        /// Value of the concrete type.
        T value{ };
    }; // struct AttributeValue

    /// @brief Get already created attribute or return nullptr.
    template <typename T>
    AttributeValue<T> *getAttribute(const KT &element, const AKT &attribute);

    /// @brief Get already created attribute or create it and return its pointer.
    template <typename T>
    AttributeValue<T> *createGetAttribute(const KT &element, const AKT &attribute);

    // TODO - Very simple and inefficient implementation.

    /// @brief Type used for storing lists of attributes for each element.
    using AttributeStorage = std::vector<std::shared_ptr<AttributeValueBase>>;

    /// Holder of data for all of the registered elements.
    std::map<KT, AttributeStorage> mElementStorage{ };
protected:
}; // class TemplateMap

/// Chain is a series of nodes forming a branch. (It's basically just a list of indeces.)
struct GraphChain
{
    /// Optional, can be used to label the chains uniquely
    unsigned int chainid;
    /// Indeces of nodes within an ArrayTree
    std::vector<int> elements;
}; // struct GraphChain

/// @brief Format given time-point time to readable string.
template <typename TimePointT>
std::string strTimestamp(const TimePointT &timePoint,
    const std::string &divider1 = "_", const std::string &divider2 = "__");

/**
 * @brief Helps you save text data without worrying about nonsesnse such as the directory existing.
 * @param filename is the path, nonexistant directory will be generated.
 * @param contents goes inside
 * @return true if saving failed.
 */
bool saveTextToFile(std::string filename, std::string contents);

/// @brief Returns the proper separator for a system.
inline const std::string sysSepStr();

} // namespace treeutil

namespace treeio
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treeio

namespace treegui
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treegui

namespace treerndr
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treerndr

namespace treert
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treert

namespace treescene
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treescene

namespace treeop
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treeop

namespace treestat
{

// Forward logging utilities:
using treeutil::Debug;
using treeutil::Info;
using treeutil::Warning;
using treeutil::Error;

// Allow simple use of Vector3D and Vector2D.
using treeutil::Vector3D;
using treeutil::Vector2D;

} // namespace treestat


// Template implementation begin.

namespace treeutil
{

inline void Serializer::reset()
{ mSs.clear(); }
inline std::string Serializer::str() const
{ return mSs.str(); }

inline void Serializer::startBlock()
{
    if (mValueWritten)
    { mSs << impl::SERIALIZER_DIVIDER; }
    mSs << impl::SERIALIZER_BLOCK_START;
    mValueWritten = false;
}
inline void Serializer::endBlock()
{ mSs << impl::SERIALIZER_BLOCK_END; mValueWritten = true; }

template <typename T>
inline void Serializer::serialize(const T &v)
{
    if (mValueWritten)
    { mSs << impl::SERIALIZER_DIVIDER; }
    mSs << v;
    mValueWritten = true;
}

template <>
inline Serializer &Serializer::operator<<(const Serializer::BlockStartT&)
{ startBlock(); return *this; }

template <>
inline Serializer &Serializer::operator<<(const Serializer::BlockEndT&)
{ endBlock(); return *this; }

template <typename T>
inline Serializer &Serializer::operator<<(const T &v)
{ serialize(v); return *this; }

inline Deserializer::Deserializer(const std::string &serialized)
{ load(serialized); }

inline void Deserializer::load(const std::string &serialized)
{ mSs = std::stringstream{ serialized }; }

inline void Deserializer::startBlock()
{
    auto readVal{ readChar(mSs) };

    if (mValueRead)
    { checkCharThrow(readVal, impl::SERIALIZER_DIVIDER); mSs >> readVal; }
    checkCharThrow(readVal, impl::SERIALIZER_BLOCK_START);

    mValueRead = false;
}
inline void Deserializer::endBlock()
{
    const auto readVal{ readChar(mSs) };

    checkCharThrow(readVal, impl::SERIALIZER_BLOCK_END);

    mValueRead = true;
}

template <typename T>
inline T Deserializer::deserialize()
{
    if (mValueRead)
    { const auto readVal{ readChar(mSs) }; checkCharThrow(readVal, impl::SERIALIZER_DIVIDER); }

    T val{ };
    mSs >> val;

    mValueRead = true;
    return val;
}

template <>
inline Deserializer &Deserializer::operator>>(const Deserializer::BlockStartT&)
{ startBlock(); return *this; }

template <>
inline Deserializer &Deserializer::operator>>(const Deserializer::BlockEndT&)
{ endBlock(); return *this; }

template <typename T>
inline Deserializer &Deserializer::operator>>(T &v)
{ v = deserialize<T>(); return *this; }

inline char Deserializer::readChar(std::stringstream &stream)
{ char readVal{ }; stream >> readVal; return readVal; }

inline void Deserializer::checkCharThrow(char readValue, char expectedValue)
{
    if (readValue != expectedValue)
    { throw DeserializerException(std::string("Expected " ) + expectedValue + " got " + readValue); }
}

template <typename KT, typename AKT>
TemplateMap<KT, AKT>::TemplateMap()
{ /* Automatic */ }
template <typename KT, typename AKT>
TemplateMap<KT, AKT>::~TemplateMap()
{ /* Automatic */ }

template <typename KT, typename AKT>
void TemplateMap<KT, AKT>::clear()
{ mElementStorage.clear(); }

template <typename KT, typename AKT>
bool TemplateMap<KT, AKT>::empty()
{ return mElementStorage.empty(); }

template <typename KT, typename AKT>
template <typename T>
bool TemplateMap<KT, AKT>::has(const KT &element, const AKT &attribute) const
{ return getAttribute<T>(element, attribute) != nullptr; }

template <typename KT, typename AKT>
template <typename T>
T &TemplateMap<KT, AKT>::get(const KT &element, const AKT &attribute) const
{ return createGetAttribute<T>(element, attribute)->value; }

template <typename KT, typename AKT>
TemplateMap<KT, AKT>::AttributeValueBase::AttributeValueBase(const AKT &attributeName) :
    name{ attributeName } { }

template <typename KT, typename AKT>
template <typename T>
typename TemplateMap<KT, AKT>::template AttributeValue<T> *TemplateMap<KT, AKT>::AttributeValueBase::specialize()
{ return dynamic_cast<AttributeValue<T>*>(this); }

template <typename KT, typename AKT>
template <typename T>
TemplateMap<KT, AKT>::AttributeValue<T>::AttributeValue(const AKT &name) :
    AttributeValueBase(name) { }

template <typename KT, typename AKT>
template <typename T>
typename TemplateMap<KT, AKT>::template AttributeValue<T> *TemplateMap<KT, AKT>::getAttribute(
    const KT &element, const AKT &attribute)
{
    // Look for the element name:
    const auto findIt{ mElementStorage.find(element) };
    if (findIt == mElementStorage.end())
    { return nullptr; }

    // Look for the attribute name:
    const auto &attributes{ findIt->second };
    for (const auto &attr: attributes)
    { if (attr.name == attribute) { return attr->template specialize<T>(); } }

    // No attribute with given name found.
    return nullptr;
}

template <typename KT, typename AKT>
template <typename T>
typename TemplateMap<KT, AKT>::template AttributeValue<T> *TemplateMap<KT, AKT>::createGetAttribute(
    const KT &element, const AKT &attribute)
{
    // Create or get the element:
    const auto [it, created]{ mElementStorage.insert(element) };

    // Look for the attribute name:
    const auto &attributes{ it->second };
    for (const auto &attr: attributes)
    { if (attr.name == attribute) { return attr->template specialize<T>(); }; }

    // No attribute with given name found -> Create it!
    const auto attributeValue{ std::make_shared<AttributeValue<T>>(attribute) };
    attributes.push_back(std::dynamic_pointer_cast<AttributeValueBase>(attributeValue));

    return attributeValue.get();
}

template <typename InstanceT, typename ReturnT, typename... ArgumentTs>
std::function<ReturnT(ArgumentTs...)> closure(InstanceT *instance, ReturnT(InstanceT::*method)(ArgumentTs...))
{ return [instance, method] (ArgumentTs... arguments) { (instance->*method)(std::forward<ArgumentTs>(arguments)...); }; }

template <typename TimePointT>
std::string strTimestamp(const TimePointT &timePoint,
    const std::string &divider1, const std::string &divider2)
{
    using ClockT = typename TimePointT::clock;
    const auto time{ ClockT::to_time_t(timePoint) };

    tm buffer{ };
#ifdef _WIN32
    // TODO - localtime_s should return nullptr on error, but it returns nullptr always?
#if 1
    localtime_s(&buffer, &time);
#else
    if (!localtime_s(&buffer, &time))
    { return ""; }
#endif
#else // _WIN32
    if (!localtime_r(&time, &buffer))
    { return ""; }
#endif // _WIN32

    std::stringstream ss{ };
    ss << buffer.tm_year + 1900u << divider1
       << buffer.tm_mon + 1u << divider1
       << buffer.tm_mday << divider2
       << buffer.tm_hour << divider1
       << buffer.tm_min << divider1
       << buffer.tm_sec;

    return ss.str();
}

/// Returns the proper separator for a system.
inline const std::string sysSepStr()
{
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}

} // namespace treeutil

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const treeutil::VersionCheckpoint<T> &version)
{ treeutil::Info << version.identifier() << "#" << version.version(); return out; }

template <typename T, std::size_t Size>
inline std::ostream &operator<<(std::ostream &out, const std::array<T, Size> &arr)
{
    out << "[ ";
    for (std::size_t iii = 0u; iii < Size; ++iii)
    {
        out << arr[iii];
        if (iii < Size - 1u)
        { out << " , "; }
    }
    out << " ]";
    return out;
}
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &arr)
{
    out << "[ ";
    for (std::size_t iii = 0u; iii < arr.size(); ++iii)
    {
        out << arr[iii];
        if (iii < arr.size() - 1u)
        { out << " , "; }
    }
    out << " ]";
    return out;
}

// @brief from: http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html
template <class Container>
inline void strsplit(const std::string& str, Container& cont,
                     char delimiter = ' ')
{
    std::size_t current{ str.find(delimiter) };
    std::size_t previous{ };
    while (current != std::string::npos)
    {
        cont.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delimiter, previous);
    }
    cont.push_back(str.substr(previous, current - previous));
}

// Template implementation end.

#endif // TREE_UTILS_H

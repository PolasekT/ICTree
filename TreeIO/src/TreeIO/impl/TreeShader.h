/**
 * @author Tomas Polasek, David Hrusa
 * @date 20.3.2020
 * @version 1.0
 * @brief Wrapper around compute shader and various utilities.
 */

#ifndef TREE_COMPUTE_SHADER_H
#define TREE_COMPUTE_SHADER_H

#include "TreeUtils.h"
#include "TreeGLUtils.h"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace treerndr
{

/// @brief Information about a shader.
struct Shader
{
    /// @brief Empty shader.
    Shader() = default;
    /// @brief Shader loaded from given filename.
    Shader(const std::string &path, GLenum stage);
    /// @brief Shader created from source code.
    Shader(GLenum stage, const std::string &source);

    /// @brief Print description of this shader.
    void describe(std::ostream &out, const std::string &indent = "") const;

    /// Filename containing shader code.
    std::string filename{ };
    /// Type of shader.
    GLenum type{ GL_VERTEX_SHADER };
    /// Source code for the shader.
    std::string source{ };
}; // struct Shader

/// @brief Information about compiled shader program.
class ShaderProgram : public treeutil::PointerWrapper<ShaderProgram>
{
public:
    /// @brief Empty shader program.
    ShaderProgram();
    /// @brief Shader program create from given list of shaders.
    ShaderProgram(std::initializer_list<Shader> stages);
    /// @brief Shader program create from given list of shaders.
    ShaderProgram(const std::vector<Shader> &stages);

    /// @brief Compile the current list of shaders. Returns true on success.
    bool compile(const std::vector<std::string> &transformOutputs = { });

    /// @brief Is this shader program compiled?
    bool compiled() const;

    /// @brief Delete the currently compiled shader program. No effect when not compiled.
    void reset();

    /// @brief Use this program. Must be already compiled! Returns true on success.
    bool use() const;

    /// @brief Get OpenGL identifier/name of the managed shader program.
    GLuint id() const;

    /// @brief Does this shader program have given uniform name?
    bool hasUniform(const char *name) const;
    /// @brief Does this shader program have given uniform name?
    bool hasUniform(const std::string &name) const;

    /// @brief Set uniform automatically based on its type. Returns success.
    template <typename T>
    bool setUniformStr(const std::string &name, const T &value) const;

    /// @brief Print description of this shader program.
    void describe(std::ostream &out, const std::string &indent = "") const;

    // Float uniforms:
    /// @brief Set uniform to given float. Returns success.
    bool setUniform(const char *name, const float &value) const;
    /// @brief Set uniform to given float. Returns success.
    bool setUniform(const char *name, const glm::vec1 &value) const;
    /// @brief Set uniform to given vec2. Returns success.
    bool setUniform(const char *name, const Vector2D &value) const;
    /// @brief Set uniform to given vec2. Returns success.
    bool setUniform(const char *name, const glm::vec2 &value) const;
    /// @brief Set uniform to given vec3. Returns success.
    bool setUniform(const char *name, const Vector3D &value) const;
    /// @brief Set uniform to given vec3. Returns success.
    bool setUniform(const char *name, const glm::vec3 &value) const;
    /// @brief Set uniform to given vec4. Returns success.
    bool setUniform(const char *name, const glm::vec4 &value) const;

    // Integer uniforms:
    /// @brief Set uniform to given int. Returns success.
    bool setUniform(const char *name, const int &value) const;
    /// @brief Set uniform to given int. Returns success.
    bool setUniform(const char *name, const glm::ivec1 &value) const;
    /// @brief Set uniform to given ivec2. Returns success.
    bool setUniform(const char *name, const glm::ivec2 &value) const;
    /// @brief Set uniform to given ivec3. Returns success.
    bool setUniform(const char *name, const glm::ivec3 &value) const;
    /// @brief Set uniform to given ivec4. Returns success.
    bool setUniform(const char *name, const glm::ivec4 &value) const;

    // Float array uniforms:
    /// @brief Set uniform to given float. Returns success.
    bool setUniform(const char *name, const float *value, std::size_t count) const;
    /// @brief Set uniform to given float. Returns success.
    bool setUniform(const char *name, const glm::vec1 *value, std::size_t count) const;
    /// @brief Set uniform to given vec2. Returns success.
    bool setUniform(const char *name, const Vector2D *value, std::size_t count) const;
    /// @brief Set uniform to given vec2. Returns success.
    bool setUniform(const char *name, const glm::vec2 *value, std::size_t count) const;
    /// @brief Set uniform to given vec3. Returns success.
    bool setUniform(const char *name, const Vector3D *value, std::size_t count) const;
    /// @brief Set uniform to given vec3. Returns success.
    bool setUniform(const char *name, const glm::vec3 *value, std::size_t count) const;
    /// @brief Set uniform to given vec4. Returns success.
    bool setUniform(const char *name, const glm::vec4 *value, std::size_t count) const;

    // Integer array uniforms:
    /// @brief Set uniform to given int. Returns success.
    bool setUniform(const char *name, const int *value, std::size_t count) const;
    /// @brief Set uniform to given int. Returns success.
    bool setUniform(const char *name, const glm::ivec1 *value, std::size_t count) const;
    /// @brief Set uniform to given ivec2. Returns success.
    bool setUniform(const char *name, const glm::ivec2 *value, std::size_t count) const;
    /// @brief Set uniform to given ivec3. Returns success.
    bool setUniform(const char *name, const glm::ivec3 *value, std::size_t count) const;
    /// @brief Set uniform to given ivec4. Returns success.
    bool setUniform(const char *name, const glm::ivec4 *value, std::size_t count) const;

    // Matrix uniforms:
    /// @brief Set uniform to given mat4. Returns success.
    bool setUniform(const char *name, const glm::mat4 &value) const;

    // Sampler uniforms:
    /// @brief Set uniform to given texture buffer. Returns success.
    bool setUniform(const char *name, const TextureBuffer &value, std::size_t unit = 0u) const;
    /// @brief Set uniform to given texture buffer. Returns success.
    bool setUniform(const char *name, const treeutil::WrapperPtrT<TextureBuffer> &value, std::size_t unit = 0u) const;
    /// @brief Set uniform to given texture buffer. Returns success.
    bool setUniform(const char *name, const treeutil::WrapperPtrT<TextureBuffer> *value, std::size_t count) const;
    /// @brief Set uniform to given texture buffer. Returns success.
    template <std::size_t Size>
    bool setUniform(const char *name, const std::array<treeutil::WrapperPtrT<TextureBuffer>, Size> &value,
        std::size_t unit = 0u) const;

    // Buffer uniforms:
    /// @brief Set uniform to given buffer. Returns success.
    bool setUniform(const char *name, const Buffer &value, std::size_t layout = 0u) const;
    /// @brief Set uniform to given buffer. Returns success.
    bool setUniform(const char *name, const treeutil::WrapperPtrT<Buffer> &value, std::size_t layout = 0u) const;
    /// @brief Set uniform to given buffer. Returns success.
    bool setUniform(const char *name, const treeutil::WrapperPtrT<Buffer> *value, std::size_t count) const;
    /// @brief Set uniform to given buffer. Returns success.
    template <std::size_t Size>
    bool setUniform(const char *name, const std::array<treeutil::WrapperPtrT<Buffer>, Size> &value,
        std::size_t layout = 0u) const;
private:
protected:
    /// @brief Helper structure used to hold the shader program.
    struct ProgramHolder
    {
        /// @brief Automatically destroy held program.
        ~ProgramHolder();

        /// Identifier of the compiled shader program.
        GLuint id{ };
    }; // struct ProgramHolder

    /// List of shaders making up the shader program.
    std::vector<Shader> mShaders{ };
    /// Holder for the currently managed shader program.
    std::shared_ptr<ProgramHolder> mProgram{ };
}; // struct ShaderProgram

/// @brief Functional style helper for building ShaderPrograms.
class ShaderProgramHelper
{
public:
    // Functional stage adders:
    /// @brief Add vertex shader from given path.
    ShaderProgramHelper &addVertex(const std::string &path);
    /// @brief Add vertex shader from given source code.
    ShaderProgramHelper &addVertexSource(const std::string &source);
    /// @brief Add vertex shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addVertexFallback(const std::string &path, const std::string &source);
    /// @brief Add tessellation control shader from given path.
    ShaderProgramHelper &addControl(const std::string &path);
    /// @brief Add tessellation control shader from given source code.
    ShaderProgramHelper &addControlSource(const std::string &source);
    /// @brief Add tessellation control shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addControlFallback(const std::string &path, const std::string &source);
    /// @brief Add tessellation evaluation shader from given path.
    ShaderProgramHelper &addEvaluation(const std::string &path);
    /// @brief Add tessellation evaluation shader from given source code.
    ShaderProgramHelper &addEvaluationSource(const std::string &source);
    /// @brief Add tessellation evaluation shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addEvaluationFallback(const std::string &path, const std::string &source);
    /// @brief Add geometry shader from given path.
    ShaderProgramHelper &addGeometry(const std::string &path);
    /// @brief Add geometry shader from given source code.
    ShaderProgramHelper &addGeometrySource(const std::string &source);
    /// @brief Add geometry shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addGeometryFallback(const std::string &path, const std::string &source);
    /// @brief Add fragment shader from given path.
    ShaderProgramHelper &addFragment(const std::string &path);
    /// @brief Add fragment shader from given source code.
    ShaderProgramHelper &addFragmentSource(const std::string &source);
    /// @brief Add fragment shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addFragmentFallback(const std::string &path, const std::string &source);
    /// @brief Add compute shader from given path.
    ShaderProgramHelper &addCompute(const std::string &path);
    /// @brief Add compute shader from given source code.
    ShaderProgramHelper &addComputeSource(const std::string &source);
    /// @brief Add compute shader from given source path, falling back to provided source code.
    ShaderProgramHelper &addComputeFallback(const std::string &path, const std::string &source);

    // Functional setting setters:
    /// @brief Enable or disable transform feedback. By default this is disabled.
    ShaderProgramHelper &transformFeedback(const std::vector<std::string> &attributes = { });

    // Functional checkers:
    /// @brief Check that all of given uniform names are available. Automatically compiles if necessary.
    ShaderProgramHelper &checkUniforms(std::initializer_list<std::string> uniformNames);
    /// @brief Check that all of given uniform names are available. Automatically compiles if necessary.
    ShaderProgramHelper &checkUniforms(const std::vector<std::string> &uniformNames);

    // Functional finalizers:
    /// @brief Build the current ShaderProgram and check for errors.
    ShaderProgramHelper &build();
    /// @brief Finalize the shader program, compile if necessary and check for errors.
    [[nodiscard]] ShaderProgram finalize();
    /// @brief Finalize the shader program, compile if necessary and check for errors. This version returns pointer.
    ShaderProgram::Ptr finalizePtr();
private:
    /// @brief Update program shader stages if necessary.
    void updateStages();
    /// @brief Compile the shader program if necessary and check for errors.
    void compile();

    /// Did we add or remove any shader stages?
    bool mStagesChanged{ false };
    /// List of currently added shader stages.
    std::vector<Shader> mStages{ };
    /// List of transform feedbacks outputs of this shader.
    std::vector<std::string> mTransformFeedbackOutputs{ };
    /// Do we need to recompile the shader program?
    bool mRecompilationRequired{ true };
    /// Internal program being worked on.
    ShaderProgram mProgram{ };
protected:
}; // class ShaderProgramHelper

/// @brief Helper for building shader programs.
struct ShaderProgramFactory
{
    /// Pure static factory methods only.
    ShaderProgramFactory() = delete;

    /// @brief Start building a new rendering shader program.
    static ShaderProgramHelper renderProgram();
}; // struct ShaderProgramFactory

/// @brief Holds information about a single uniform of UniformT type.
template <typename UniformT>
class UniformEntry
{
public:
    /// @brief Internal uniform type.
    using ValueT = UniformT;

    /// @brief Initialize the uniform with given values.
    template <typename... CArgTs>
    UniformEntry(const std::string &name, const std::string &identifier,
        const std::string &description, CArgTs... cArgs);
    /// @brief Clean up the internal value and destroy.
    ~UniformEntry();

    // Copy and move semantic:
    UniformEntry(const UniformEntry &other);
    UniformEntry &operator=(UniformEntry other);
    UniformEntry(UniformEntry &&other);
    UniformEntry &operator=(UniformEntry &&other);

    /// @brief Swap content of this and the other instance.
    void swap(UniformEntry &other);

    /// @brief Swap content of first and second instances.
    static void swap(UniformEntry &first, UniformEntry &second);

    /// @brief Human readable name of the uniform.
    std::string name() const;
    /// @brief String identifier of the uniform variable.
    std::string identifier() const;
    /// @brief Description of uniform content and use.
    std::string description() const;
    /// @brief OpenGL type of the uniform.
    GLenum type() const;
    /// @brief Value of the uniform.
    const ValueT &value() const;
    /// @brief Value of the uniform.
    ValueT &value();
    /// @brief Unit used by this uniform.
    std::size_t unit() const;
    /// @brief Layout used by this uniform.
    std::size_t layout() const;

    /// @brief Convert to underlying value.
    operator const ValueT&() const;
    /// @brief Convert to underlying value.
    operator ValueT &();

    /// @brief Convert to underlying value.
    UniformEntry &operator=(const ValueT &val);

    /// @brief Mimic operator[].
    template <typename T>
    auto &operator[](const T &idx) { return value()[idx]; }

    /// @brief Mimic operator[].
    template <typename T>
    const auto &operator[](const T &idx) const { return value()[idx]; }

    /// @brief Set uniform value for given shader program.
    bool setUniform(const ShaderProgram &program) const;

    /// @brief Print description of this uniform entry.
    void describe(std::ostream &out, const std::string &indent = "") const;
private:
    /// @brief Holder for the internal value.
    template <typename ValT, typename ET = void>
    struct ValueHolder;

    /// Human readable name of the uniform.
    std::string mName{ };
    /// String identifier of the uniform variable.
    std::string mIdentifier{ };
    /// Description of uniform content and use.
    std::string mDescription{ };
    /// OpenGL type of the uniform.
    GLenum mType{ };
    /// Value of the uniform.
    ValueHolder<UniformT> mValue{ };
protected:
}; // class UniformEntry

/// @brief Convert OpenGL type into C++ type - e.g. mat4 into glm::mat4.
#define gl2cpp(glType) \
    treeutil::glt::glType##T::type

/// @brief Construct array type with given OpenGL type and size and translate it into C++ type.
#define glarray(glType, size) \
    treeutil::glt::arrayT<gl2cpp(glType), size>::type

/// @brief Helper for uniform macro initializer construction.
#define uniform_inner_initial(identifier, description, initial) \
    { #identifier, #identifier, description, initial }
/// @brief Helper for uniform macro initializer construction.
#define uniform_inner_no_initial(identifier, description) \
    { #identifier, #identifier, description }
/// @brief Helper used for getting correct uniform macro.
#define uniform_macro_getter(_1, _2, _3, name, ...) name
/**
 * @brief Define uniform compatibility variable using type, identifier, description and optionally initializer.
 *
 * @param glType Type of the uniform as defined in GLSL - e.g. mat4, vec3, ... .
 * @param identifier Identifier of the uniform as defined in GLSL - e.g. uMVP, uLightPos, ... .
 * @param description String containing uniform description.
 * @param initial Initial value of the uniform. In case of sampler uniforms this parameter
 *  has the semantic of texture unit index, which defaults to unit 0. This is optional and
 *  may be omitted.
 *
 * @usage:
 *  uniform(mat4, uMVP, "My model-view-projection matrix.");
 *  uniform(bool, uUseShadows, "Set to true to enable shadows", true);
 *  ...
 *  uMVP = glm::mat4(1.0f);
 *  // uUseShadows is true by default.
 */
#define uniform(glType, identifier, ...) \
    UniformEntry<gl2cpp(glType)> identifier \
    uniform_macro_getter(identifier, __VA_ARGS__, uniform_inner_initial, uniform_inner_no_initial)(identifier, __VA_ARGS__)

/**
 * @brief Define array uniform compatibility variable using type, size, identifier, description and optionally initializer.
 *
 * @param glType Type of the uniform as defined in GLSL - e.g. mat4, vec3, ... .
 * @param size Size of the array - e.g. 13, 42, ... .
 * @param identifier Identifier of the uniform as defined in GLSL - e.g. uMVP, uLightPos, ... .
 * @param description String containing uniform description.
 * @param initial Initial value of the uniform. In case of sampler uniforms this parameter
 *  has the semantic of texture unit index, which defaults to unit 0. This is optional and
 *  may be omitted.
 *
 * @usage:
 *  uniformArr(mat4, 4, uMVPs, "My model-view-projection matrix.");
 *  ...
 *  uMVP = { glm::mat4(1.0f) };
 */
#define uniformArr(glType, size, identifier, ...) \
    UniformEntry<glarray(glType, size)> identifier \
    uniform_macro_getter(identifier, __VA_ARGS__, uniform_inner_initial, uniform_inner_no_initial)(identifier, __VA_ARGS__)

/// @brief Set of uniform variables, their values and other settings.
class UniformSet
{
public:
    /// @brief Initialize empty set of uniform variables.
    UniformSet();
    /// @brief Initialize the set from provided uniform entries. The values can be kept by reference or copied inside.
    template <typename... UniformTs>
    UniformSet(bool copy, UniformEntry<UniformTs>&... entries);

    /// @brief Add given uniform to this set. The value can be kept by reference or copied inside.
    template <typename UniformT>
    void addUniform(bool copy, UniformEntry<UniformT> &entry);
    /// @brief Add given uniforms to this set. The value can be kept by reference or copied inside. Variadic iterator.
    template <typename UniformT, typename... UniformTs>
    void addUniforms(bool copy, UniformEntry<UniformT> &entry, UniformEntry<UniformTs>&... entries);
    /// @brief Add given uniforms to this set. The value can be kept by reference or copied inside. End of iteration.
    inline void addUniforms(bool copy);

    /// @brief Add uniform to this set by providing its parameters.
    template <typename UniformT>
    void addUniform(const std::string &name, const std::string &identifier,
        const std::string &description, const UniformT &initial = { });

    /// @brief Access uniform by identifier.
    template <typename UniformT>
    UniformEntry<UniformT> &getUniform(const std::string &identifier);
    /// @brief Access uniform by identifier.
    template <typename UniformT>
    const UniformEntry<UniformT> &getUniform(const std::string &identifier) const;

    /// @brief Set all uniform contained within this set for given program. Returns number of set uniforms.
    std::size_t setUniforms(const ShaderProgram &program) const;

    /// @brief Get list of uniform identifiers containing every registered uniform entry.
    std::vector<std::string> getUniformIdentifiers() const;

    /// @brief Print description of this uniform set.
    void describe(std::ostream &out, const std::string &indent = "") const;
private:
    /// @brief Helper for keeping various uniform types with constant interface.
    struct UniformEntryWrapperBase
    {
        /// @brief Clean up and destroy any internal data.
        virtual ~UniformEntryWrapperBase() = default;
        /// @brief Set uniform value for given program.
        virtual bool setUniform(const ShaderProgram &program) const = 0;
        /// @brief Print description of the internal uniform entry.
        virtual void describe(std::ostream &out, const std::string &indent = "") const = 0;
        /// @brief Get internal pointer of the uniform entry.
        virtual void *getUniformEntryPtr() = 0;
    }; // struct UniformEntryWrapperBase

    template <typename UniformT>
    struct UniformDelete
    {
        /// Pointer to the underlying UniformEntry.
        using UniformEntryT = UniformEntry<UniformT>;
        /// @brief Delete interface.
        void operator()(UniformEntryT *p)
        { deleteFun(p); }
        /// Function used when deleting. By default uses behavior of std::default_delete.
        std::function<void(UniformEntryT*)> deleteFun{
            [] (UniformEntryT *p) { std::default_delete<UniformEntryT>()(p); }
        };
    }; // struct UniformDelete

    /// @brief Helper for keeping various uniform types with constant interface.
    template <typename UniformT>
    struct UniformEntryWrapper : public UniformEntryWrapperBase
    {
        /// @brief Clean up and destroy any internal data.
        virtual ~UniformEntryWrapper();
        /// @brief Set uniform value for given program.
        virtual bool setUniform(const ShaderProgram &program) const override final;
        /// @brief Print description of the internal uniform entry.
        virtual void describe(std::ostream &out, const std::string &indent = "") const override final;
        /// @brief Get internal pointer of the uniform entry.
        virtual void *getUniformEntryPtr() override final;

        /// Pointer to the internal uniform data.
        std::unique_ptr<UniformEntry<UniformT>, UniformDelete<UniformT>> uniformPtr{ };
    }; // struct UniformEntryWrapper

    /// @brief Add given uniform entry by copying it into the set.
    template <typename UniformT>
    void addUniformCopy(const UniformEntry<UniformT> &entry);
    /// @brief Add given uniform entry by adding its reference to the set.
    template <typename UniformT>
    void addUniformReference(UniformEntry<UniformT> &entry);

    /// Mapping from uniform identifier to its record.
    std::map<std::string, std::shared_ptr<UniformEntryWrapperBase>> mUniformMap{ };
protected:
}; // class UniformSet

} // namespace treerndr

/// @brief Print shader description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::Shader &shader);
/// @brief Print shader program description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderProgram &shaderProgram);
/// @brief Print uniform description.
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const treerndr::UniformEntry<T> &uniformEntry);
/// @brief Print uniform set description.
inline std::ostream &operator<<(std::ostream &out, const treerndr::UniformSet &uniformSet);

// Template implementation begin.

namespace treerndr
{
template <std::size_t Size>
bool ShaderProgram::setUniform(const char *name,
    const std::array<treeutil::WrapperPtrT<TextureBuffer>, Size> &value, std::size_t unit) const
{
    auto baseUnit{ unit };
    auto success{ true };
    for (const auto &v : value)
    { success = success && setUniform(name, v, baseUnit++); }
    return success;
}

template <std::size_t Size>
bool ShaderProgram::setUniform(const char *name,
    const std::array<treeutil::WrapperPtrT<Buffer>, Size> &value, std::size_t layout) const
{
    auto baseLayout{ layout };
    auto success{ true };
    for (const auto &v : value)
    { success = success && setUniform(name, v, baseLayout++); }
    return success;
}

template <typename T>
bool ShaderProgram::setUniformStr(const std::string &name, const T &value) const
{ return setUniform(name.c_str(), value); }

template <typename UniformT>
template <typename... CArgTs>
UniformEntry<UniformT>::UniformEntry(const std::string &name, const std::string &identifier,
    const std::string &description, CArgTs... cArgs) :
    mName{ name }, mIdentifier{ identifier }, mDescription{ description },
    mType{ treeutil::typeToGLType<UniformT>() }, mValue{ std::forward<CArgTs>(cArgs)... }
{ }
template <typename UniformT>
UniformEntry<UniformT>::~UniformEntry()
{ /* Automatic */ }

template <typename UniformT>
UniformEntry<UniformT>::UniformEntry(const UniformEntry &other) :
    mName{ other.mName }, mIdentifier{ other.mIdentifier }, mDescription{ other.mDescription },
    mType{ other.mType }, mValue{ other.mValue }
{ }
template <typename UniformT>
UniformEntry<UniformT> &UniformEntry<UniformT>::operator=(UniformEntry other)
{ swap(*this, other); return *this; }
template <typename UniformT>
UniformEntry<UniformT>::UniformEntry(UniformEntry &&other)
{ swap(*this, other); }
template <typename UniformT>
UniformEntry<UniformT> &UniformEntry<UniformT>::operator=(UniformEntry &&other)
{ swap(*this, other); return *this; }

template <typename UniformT>
void UniformEntry<UniformT>::swap(UniformEntry &other)
{ swap(*this, other); }

template <typename UniformT>
void UniformEntry<UniformT>::swap(UniformEntry &first, UniformEntry &second)
{
    std::swap(first.mName, second.mName);
    std::swap(first.mIdentifier, second.mIdentifier);
    std::swap(first.mDescription, second.mDescription);
    std::swap(first.mType, second.mType);
    std::swap(first.mValue, second.mValue);
}

template <typename UniformT>
std::string UniformEntry<UniformT>::name() const
{ return mName; }
template <typename UniformT>
std::string UniformEntry<UniformT>::identifier() const
{ return mIdentifier; }
template <typename UniformT>
std::string UniformEntry<UniformT>::description() const
{ return mDescription; }
template <typename UniformT>
GLenum UniformEntry<UniformT>::type() const
{ return mType; }
template <typename UniformT>
const UniformT &UniformEntry<UniformT>::value() const
{ return mValue.v; }
template <typename UniformT>
UniformT &UniformEntry<UniformT>::value()
{ return mValue.v; }
template <typename UniformT>
std::size_t UniformEntry<UniformT>::unit() const
{ return mValue.u; }
template <typename UniformT>
std::size_t UniformEntry<UniformT>::layout() const
{ return mValue.l; }

template <typename UniformT>
UniformEntry<UniformT>::operator const UniformT&() const
{ return mValue.v; }
template <typename UniformT>
UniformEntry<UniformT>::operator UniformT&()
{ return mValue.v; }

template <typename UniformT>
UniformEntry<UniformT> &UniformEntry<UniformT>::operator=(const UniformT &val)
{ mValue.v = val; return *this; }

template <typename UniformT>
bool UniformEntry<UniformT>::setUniform(const ShaderProgram &program) const
{ return mValue.setUniform(program, mIdentifier); }

template <typename UniformT>
void UniformEntry<UniformT>::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ UniformEntry: \n"
        << indent << "\tName = " << mName << "\n"
        << indent << "\tIdentifier = " << mIdentifier << "\n"
        << indent << "\tDescription = " << mDescription << "\n"
        << indent << "\tType = " << treeutil::glTypeToStr(mType) << "\n"
        << indent << "\tValue = ";
    mValue.describe(out, indent + "\t");
    out << "\n" << indent << " ]";
}

// Default case: Contains the value itself.
template <typename UniformT>
template <typename ValT, typename ET>
struct UniformEntry<UniformT>::ValueHolder
{
    // Mimic initialization of the underlying type.
    ValueHolder() = default;
    template <typename... CArgTs>
    ValueHolder(CArgTs... cArgs) :
        v(std::forward<CArgTs>(cArgs)...)
    { }

    /// @brief Set uniform for given shader program.
    bool setUniform(const ShaderProgram &program, const std::string &identifier) const
    { return program.setUniform(identifier.c_str(), v); }

    /// @brief Print description of this value.
    void describe(std::ostream &out, const std::string &indent = "") const
    { out << "[ ValueHolder: v = " << v << " ]"; }

    /// Underlying value.
    ValT v{ };
}; // struct ValueHolder

// Texture types: Contains the texture and its texture unit identifier.
template <typename UniformT>
template <typename ValT>
struct UniformEntry<UniformT>::ValueHolder<ValT, typename std::enable_if<treeutil::typeIsTexture<ValT>::value>::type>
{
    // Mimic initialization of the underlying type with texture unit taking priority.
    ValueHolder() = default;
    template <typename... CArgTs>
    ValueHolder(std::size_t unit, CArgTs... cArgs) :
        u{ unit }, v(std::forward<CArgTs>(cArgs)...)
    { }

    /// @brief Activate the texture for given shader program.
    bool setUniform(const ShaderProgram &program, const std::string &identifier) const
    { return v ? program.setUniform(identifier.c_str(), *v, u) : false; }

    /// @brief Print description of this value.
    void describe(std::ostream &out, const std::string &indent = "") const
    { out << "[ ValueHolder: Value = "; if(v) out << *v; else out << "Empty"; out << " | Unit = " << u << " ]"; }

    /// Texture unit to bind to.
    std::size_t u{ };
    /// Underlying texture.
    ValT v{ };
}; // struct ValueHolder

// Buffer types: Contains the buffer and the layout location.
template <typename UniformT>
template <typename ValT>
struct UniformEntry<UniformT>::ValueHolder<ValT, typename std::enable_if<treeutil::typeIsBuffer<ValT>::value>::type>
{
    // Mimic initialization of the underlying type with texture unit taking priority.
    ValueHolder() = default;
    template <typename... CArgTs>
    ValueHolder(std::size_t layout, CArgTs... cArgs) :
        l{ layout }, v(std::forward<CArgTs>(cArgs)...)
    { }

    /// @brief Activate the buffer for given shader program.
    bool setUniform(const ShaderProgram &program, const std::string &identifier) const
    { return v ? program.setUniform(identifier.c_str(), *v, l) : false; }

    /// @brief Print description of this value.
    void describe(std::ostream &out, const std::string &indent = "") const
    { out << "[ ValueHolder: Value = "; if (v) out << *v; else out << "Empty"; out << " | Layout = " << l << " ]"; }

    /// Layout to bind to.
    std::size_t l{ };
    /// Underlying texture.
    ValT v{ };
}; // struct ValueHolder

// Array types: Contains the array of values.
template <typename UniformT>
template <typename ValT>
struct UniformEntry<UniformT>::ValueHolder<ValT, typename std::enable_if<treeutil::is_iterable<ValT>::value>::type>
{
    // Mimic initialization of the underlying type with texture unit taking priority.
    ValueHolder() = default;
    template <typename... CArgTs>
    ValueHolder(CArgTs... cArgs) :
        v(std::forward<CArgTs>(cArgs)...)
    { }

    /// @brief Activate the buffer for given shader program.
    bool setUniform(const ShaderProgram &program, const std::string &identifier) const
    { return program.setUniform(identifier.c_str(), &v[0], v.size()); }

    /// @brief Print description of this value.
    void describe(std::ostream &out, const std::string &indent = "") const
    { out << "[ ValueHolder: Value = " << v << " | Size = " << v.size() << " ]"; }

    /// Underlying texture.
    ValT v{ };
}; // struct ValueHolder

template <typename... UniformTs>
UniformSet::UniformSet(bool copy, UniformEntry<UniformTs>&... entries)
{ addUniforms(copy, entries...); }

template <typename UniformT>
void UniformSet::addUniform(bool copy, UniformEntry<UniformT> &entry)
{
    if (copy)
    { addUniformCopy(entry); }
    else
    { addUniformReference(entry); }
}
template <typename UniformT, typename... UniformTs>
void UniformSet::addUniforms(bool copy, UniformEntry<UniformT> &entry, UniformEntry<UniformTs>&... entries)
{ addUniform(copy, entry); addUniforms(copy, entries...); }
inline void UniformSet::addUniforms(bool copy)
{ /* End of variadic iteration. */ }

template <typename UniformT>
void UniformSet::addUniform(const std::string &name, const std::string &identifier,
    const std::string &description, const UniformT &initial)
{ addUniformCopy(UniformEntry(name, identifier, description, initial)); }

template <typename UniformT>
UniformEntry<UniformT> &UniformSet::getUniform(const std::string &identifier)
{
    const auto findIt{ mUniformMap.find(identifier) };
    if (findIt == mUniformMap.end())
    { throw std::runtime_error("Unable to getUniform: Unknown identifier!"); }
    const auto basePtr{ findIt->second.get() }; TREE_UNUSED(basePtr);
    const auto ptr{ std::dynamic_pointer_cast<UniformEntryWrapper<UniformT>>(findIt->second) };
    if (!ptr || !ptr->uniformPtr)
    { throw std::runtime_error("Unable to getUniform: Invalid entry, uniform of different type!"); }
    return *ptr->uniformPtr;
}
template <typename UniformT>
const UniformEntry<UniformT> &UniformSet::getUniform(const std::string &identifier) const
{
    const auto findIt{ mUniformMap.find(identifier) };
    if (findIt == mUniformMap.end())
    { throw std::runtime_error("Unable to getUniform: Unknown identifier!"); }
    const auto ptr{ std::dynamic_pointer_cast<UniformEntryWrapper<UniformT>>(findIt->second) };
    if (!ptr || !ptr->uniformPtr)
    { throw std::runtime_error("Unable to getUniform: Invalid entry, uniform of different type!"); }
    return *ptr->uniformPtr;
}

template <typename UniformT>
UniformSet::UniformEntryWrapper<UniformT>::~UniformEntryWrapper()
{ /* Automatic */ }
template <typename UniformT>
bool UniformSet::UniformEntryWrapper<UniformT>::setUniform(const ShaderProgram &program) const
{ return uniformPtr->setUniform(program); }
template <typename UniformT>
void UniformSet::UniformEntryWrapper<UniformT>::describe(std::ostream &out, const std::string &indent) const
{ uniformPtr->describe(out, indent); }
template <typename UniformT>
void *UniformSet::UniformEntryWrapper<UniformT>::getUniformEntryPtr()
{ return static_cast<void*>(uniformPtr.get()); }

template <typename UniformT>
void UniformSet::addUniformCopy(const UniformEntry<UniformT> &entry)
{
    const auto [it, inserted]{ mUniformMap.emplace(entry.identifier(), nullptr) };
    if (!inserted)
    { throw std::runtime_error("Unable to addUniformCopy: Duplicate identifier!"); }
    auto wrapper{ std::make_shared<UniformEntryWrapper<UniformT>>() };
    wrapper->uniformPtr = std::unique_ptr<UniformEntry<UniformT>, UniformDelete<UniformT>>(
        new UniformEntry<UniformT>(entry)
    );
    it->second = std::static_pointer_cast<UniformEntryWrapperBase>(wrapper);
}
template <typename UniformT>
void UniformSet::addUniformReference(UniformEntry<UniformT> &entry)
{
    const auto [it, inserted]{ mUniformMap.emplace(entry.identifier(), nullptr) };
    if (!inserted)
    { throw std::runtime_error("Unable to addUniformReference: Duplicate identifier!"); }
    auto wrapper{ std::make_shared<UniformEntryWrapper<UniformT>>() };
    wrapper->uniformPtr = std::unique_ptr<UniformEntry<UniformT>, UniformDelete<UniformT>>(
        &entry, UniformDelete<UniformT>{ [] (UniformEntry<UniformT> *ptr)
        { /* Do nothing, we do not own the pointer... */ } }
    );
    it->second = std::static_pointer_cast<UniformEntryWrapperBase>(wrapper);
}

} // namespace treerndr

inline std::ostream &operator<<(std::ostream &out, const treerndr::Shader &shader)
{ shader.describe(out); return out; }
inline std::ostream &operator<<(std::ostream &out, const treerndr::ShaderProgram &shaderProgram)
{ shaderProgram.describe(out); return out; }
template <typename T>
inline std::ostream &operator<<(std::ostream &out, const treerndr::UniformEntry<T> &uniformEntry)
{ uniformEntry.describe(out); return out; }
inline std::ostream &operator<<(std::ostream &out, const treerndr::UniformSet &uniformSet)
{ uniformSet.describe(out); return out; }

// Template implementation end.

#endif // TREE_COMPUTE_SHADER_H

/**
 * @author Tomas Polasek, David Hrusa
 * @date 20.3.2020
 * @version 1.0
 * @brief Wrapper around compute shader and various utilities.
 */

#include "TreeShader.h"

#include "TreeBuffer.h"
#include "TreeTextureBuffer.h"

namespace treerndr
{

/// @brief Read shader source code from file and performing some pre-processing.
std::string readShaderFromFile(const std::string &filename,
    const std::string &includeStr = "#include ", std::size_t includeLimit = 50u)
{
    auto shaderSource{ treeutil::readWholeFile(filename) };

    // Check for #include directives and replace them with content.
    const auto baseIncludePath{ treeutil::filePath(filename) + "/" };
    // List of already included files to prevent duplicates.
    std::set<std::string> includedFiles{ };
    std::size_t includePos{ shaderSource.find(includeStr, 0u) };
    std::size_t includesFound{ 0u };
    while (includePos != shaderSource.npos && includesFound < includeLimit)
    { // Keep replacing include directives until we process all of them.

        // Locate include filename specification.
        const auto includeStartPos{ includePos };
        const auto includeFilenameStart{ includePos + includeStr.size() };
        const auto includeFilenameEnd{ shaderSource.find("\"", includeFilenameStart + 1u) };

        if (shaderSource[includeFilenameStart] != '\"' || includeFilenameEnd == shaderSource.npos)
        { Error << "Failed to find filename parsing shader include directive!" << std::endl; break; }

        // Get filename to include.
        const auto includeFilename{ shaderSource.substr(
            includeFilenameStart + 1u,
            includeFilenameEnd - includeFilenameStart - 1u
        ) };

        // Check if we already included this file.
        const auto alreadyIncluded{ includedFiles.find(includeFilename) != includedFiles.end() };

        // Get file content if we didn't include it already.
        const auto includeFileContent{
            alreadyIncluded ?
                std::string{ } :
                treeutil::readWholeFile(baseIncludePath + includeFilename)
        };

        // Replace include with file content.
        shaderSource.replace(
            includeStartPos, includeFilenameEnd - includeStartPos + 1u,
            includeFileContent
        );

        includesFound++;
        includePos = shaderSource.find(includeStr, includePos);
        includedFiles.emplace(includeFilename);
    }

    if (includesFound >= includeLimit)
    {
        Error << "Failed to load whole shader source code (\"" << filename
              << "\"), too many includes!" << std::endl;
    }

    return shaderSource;
}

Shader::Shader(const std::string &path, GLenum stage) :
    filename{ path }, type{ stage }, source{ readShaderFromFile(filename) }
{ }

Shader::Shader(GLenum stage, const std::string &source) :
    filename{ }, type{ stage }, source{ source }
{ }

void Shader::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ Shader: \n"
        << indent << "\tType = " << treeutil::shaderTypeToStr(type) << "\n"
        << indent << "\tSource Path = " << (filename.empty() ? "NONE" : filename) << "\n"
        << indent << "\tLoaded = " << (source.empty() ? "no\n" : "yes\n")
        << indent << "\tSource Length = " << source.size() << "\n"
        << indent << " ]";
}

ShaderProgram::ShaderProgram()
{ reset(); }

ShaderProgram::ShaderProgram(std::initializer_list<Shader> stages) :
    mShaders{ stages }
{ reset(); }

ShaderProgram::ShaderProgram(const std::vector<Shader> &stages) :
    mShaders{ stages }
{ reset(); }

bool ShaderProgram::compile(const std::vector<std::string> &transformOutputs)
{
    const auto prog{ glCreateProgram() };
    auto compilationFailed{ false };

    for (const auto &shader : mShaders)
    { // Compile each shader and add it to the shader program.
        const auto compiledShader{ glCreateShader(shader.type) };
        const auto sourcePtr{ shader.source.c_str() };
        /// Differences in the API require this type to be in this format.
        const auto sourcePtrPtr{ const_cast<const GLchar**>(&sourcePtr) };
        glShaderSource(compiledShader, 1u, sourcePtrPtr, nullptr);
        glCompileShader(compiledShader);

        GLint compileStatus{ };
        glGetShaderiv(compiledShader, GL_COMPILE_STATUS, &compileStatus);
        if (!compileStatus)
        {
            Error << "Failed to compile " << treeutil::shaderTypeToStr(shader.type)
                  << " shader from: \"" << shader.filename << "\": " << std::endl;
            treeutil::printShaderCompileError(Error, compiledShader);
            treeutil::printAnnotatedShaderSource(Error, shader.filename, shader.type, shader.source);
            compilationFailed = true;
        }

        glAttachShader(prog, compiledShader);
        treeutil::checkGLError("glAttachShader");
    }

    if (transformOutputs.size() != 0u)
    {
        Info << "Capturing transform feedback outputs: [ ";

        std::vector<const char*> capturedAttributes{ };
        for (const auto &attribute : transformOutputs)
        { capturedAttributes.push_back(attribute.c_str()); Info << attribute << " , "; }

        Info << " ]" << std::endl;

        glTransformFeedbackVaryings(prog,
            capturedAttributes.size(), capturedAttributes.data(),
            GL_INTERLEAVED_ATTRIBS);
    }

    treeutil::checkGLError("shader::compile before link");

    glLinkProgram(prog);

    GLint linkStatus{ };
    glGetProgramiv(prog, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus)
    {
        Error << "Failed to link the shader program: " << std::endl;
        Error << "\tShader sources: [ ";
        for (const auto &shader : mShaders)
        { Error << shader.filename << " "; }
        Error << "]" << std::endl;
        treeutil::printProgramLinkError(Error, prog);
        compilationFailed = true;
    }

    if (compilationFailed)
    { return false; }
    else
    { reset(); mProgram->id = prog; return true; }
}

bool ShaderProgram::compiled() const
{ return mProgram != 0u; }

void ShaderProgram::reset()
{ mProgram = std::make_shared<ProgramHolder>(); }

bool ShaderProgram::use() const
{
    if (mProgram)
    { glUseProgram(mProgram->id); return true; }
    else
    { return false; }
}

GLuint ShaderProgram::id() const
{ return mProgram->id; }

bool ShaderProgram::hasUniform(const char *name) const
{ return glGetUniformLocation(mProgram->id, name) >= 0; }
bool ShaderProgram::hasUniform(const std::string &name) const
{ return hasUniform(name.c_str()); }

void ShaderProgram::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ ShaderProgram: \n"
        << indent << "\tStage Count = " << mShaders.size() << "\n"
        << indent << "\tShader Stages = " << (mShaders.empty() ? "NONE\n" : "\n");
    for (std::size_t iii = 0u; iii < mShaders.size(); ++iii)
    {
        out << indent << "\t\t" << iii << " : ";
        mShaders[iii].describe(out, indent + "\t\t") ;
        out << "\n";
    }
    out << indent << "\tCompiled = " << (mProgram ? "yes\n" : "no\n")
        << indent << " ]";
}

bool ShaderProgram::setUniform(const char *name, const float &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1f(loc, value); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec1 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1f(loc, value[0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const Vector2D &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2f(loc, value[0], value[1]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec2 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2f(loc, value[0], value[1]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const Vector3D &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3f(loc, value[0], value[1], value[2]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec3 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3f(loc, value[0], value[1], value[2]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec4 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform4f(loc, value[0], value[1], value[2], value[3]); } return loc >= 0; }

bool ShaderProgram::setUniform(const char *name, const int &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1i(loc, value); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec1 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1i(loc, value[0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec2 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2i(loc, value[0], value[1]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec3 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3i(loc, value[0], value[1], value[2]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec4 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform4i(loc, value[0], value[1], value[2], value[3]); } return loc >= 0; }

bool ShaderProgram::setUniform(const char *name, const float *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1fv(loc, static_cast<GLsizei>(count), value); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec1 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const Vector2D *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec2 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const Vector3D *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec3 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::vec4 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform4fv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }

bool ShaderProgram::setUniform(const char *name, const int *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1iv(loc, static_cast<GLsizei>(count), value); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec1 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform1iv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec2 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform2iv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec3 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform3iv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }
bool ShaderProgram::setUniform(const char *name, const glm::ivec4 *value, std::size_t count) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniform4iv(loc, static_cast<GLsizei>(count), &value[0][0]); } return loc >= 0; }

bool ShaderProgram::setUniform(const char *name, const glm::mat4 &value) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { glUniformMatrix4fv(loc, 1u, false, &value[0][0]); } return loc >= 0; }

bool ShaderProgram::setUniform(const char *name, const TextureBuffer &value, std::size_t unit) const
{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { return value.bind(loc, unit); } return false; }
bool ShaderProgram::setUniform(const char *name, const treeutil::WrapperPtrT<TextureBuffer> &value, std::size_t unit) const
{ return value ? setUniform(name, *value, unit) : false; }
bool ShaderProgram::setUniform(const char *name, const treeutil::WrapperPtrT<TextureBuffer> *value, std::size_t count) const
{
    std::size_t baseUnit{ 0u };
    auto success{ true };
    for (auto it = value; it != value + count; ++it)
    {
        const auto uniformName{ std::string(name) + "[" + std::to_string(baseUnit) + "]" };
        success = success && setUniform(uniformName.c_str(), *it, baseUnit++);
    }
    return success;
}

bool ShaderProgram::setUniform(const char *name, const Buffer &value, std::size_t layout) const
//{ const auto loc{ glGetUniformLocation(mProgram->id, name) }; if (loc >= 0) { return value.bind(layout); } return false; }
{ return value.bind(layout); }
bool ShaderProgram::setUniform(const char *name, const treeutil::WrapperPtrT<Buffer> &value, std::size_t layout) const
{ return value ? setUniform(name, *value, layout) : false; }
bool ShaderProgram::setUniform(const char *name, const treeutil::WrapperPtrT<Buffer> *value, std::size_t count) const
{
    std::size_t baseLayout{ 0u };
    auto success{ true };
    for (auto it = value; it != value + count; ++it)
    {
        const auto uniformName{ std::string(name) + "[" + std::to_string(baseLayout) + "]" };
        success = success && setUniform(uniformName.c_str(), *it, baseLayout++);
    }
    return success;
}

ShaderProgram::ProgramHolder::~ProgramHolder()
{ if (id) { glDeleteProgram(id); id = { }; } }

ShaderProgramHelper &ShaderProgramHelper::addVertex(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_VERTEX_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addVertexSource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_VERTEX_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addVertexFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_VERTEX_SHADER } : Shader{ GL_VERTEX_SHADER, source }
    );
    mStagesChanged = true; return *this;
}
ShaderProgramHelper &ShaderProgramHelper::addControl(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_TESS_CONTROL_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addControlSource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_TESS_CONTROL_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addControlFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_TESS_CONTROL_SHADER } : Shader{ GL_TESS_CONTROL_SHADER, source }
    );
    mStagesChanged = true; return *this;
}
ShaderProgramHelper &ShaderProgramHelper::addEvaluation(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_TESS_EVALUATION_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addEvaluationSource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_TESS_EVALUATION_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addEvaluationFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_TESS_EVALUATION_SHADER } : Shader{ GL_TESS_EVALUATION_SHADER, source }
    );
    mStagesChanged = true; return *this;
}
ShaderProgramHelper &ShaderProgramHelper::addGeometry(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_GEOMETRY_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addGeometrySource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_GEOMETRY_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addGeometryFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_GEOMETRY_SHADER} : Shader{ GL_GEOMETRY_SHADER, source }
    );
    mStagesChanged = true; return *this;
}
ShaderProgramHelper &ShaderProgramHelper::addFragment(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_FRAGMENT_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addFragmentSource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_FRAGMENT_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addFragmentFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_FRAGMENT_SHADER} : Shader{ GL_FRAGMENT_SHADER, source }
    );
    mStagesChanged = true; return *this;
}
ShaderProgramHelper &ShaderProgramHelper::addCompute(const std::string &path)
{ mStages.emplace_back(Shader{ path, GL_COMPUTE_SHADER }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addComputeSource(const std::string &source)
{ mStages.emplace_back(Shader{ GL_COMPUTE_SHADER, source }); mStagesChanged = true; return *this; }
ShaderProgramHelper &ShaderProgramHelper::addComputeFallback(const std::string &path, const std::string &source)
{
    mStages.emplace_back(treeutil::fileExists(path) ?
        Shader{ path, GL_COMPUTE_SHADER} : Shader{ GL_COMPUTE_SHADER, source }
    );
    mStagesChanged = true; return *this;
}

ShaderProgramHelper &ShaderProgramHelper::transformFeedback(const std::vector<std::string> &attributes)
{ mTransformFeedbackOutputs = attributes; mRecompilationRequired = true; return *this; }

ShaderProgramHelper &ShaderProgramHelper::checkUniforms(std::initializer_list<std::string> uniformNames)
{
    compile();
    for (const auto &name : uniformNames)
    {
        if (!mProgram.hasUniform(name))
        { Warning << "Shader program is missing uniform named: \"" << name << "\"!" << std::endl; }
    }
    return *this;
}

ShaderProgramHelper &ShaderProgramHelper::checkUniforms(const std::vector<std::string> &uniformNames)
{
    compile();
    for (const auto &name : uniformNames)
    {
        if (!mProgram.hasUniform(name))
        { Warning << "Shader program is missing uniform named: \"" << name << "\"!" << std::endl; }
    }
    return *this;
}

ShaderProgramHelper &ShaderProgramHelper::build()
{ compile(); return *this; }
ShaderProgram ShaderProgramHelper::finalize()
{
    compile();
    const auto tmp{ mProgram };

    mRecompilationRequired = true;
    mStagesChanged = true;

    mProgram = { };

    return tmp;
}
ShaderProgram::Ptr ShaderProgramHelper::finalizePtr()
{ return ShaderProgram::instantiate(finalize()); }

void ShaderProgramHelper::updateStages()
{
    if (!mStagesChanged)
    { return; }

    mProgram = ShaderProgram(mStages);

    mStagesChanged = false;
    mRecompilationRequired = true;
}
void ShaderProgramHelper::compile()
{
    updateStages();
    if (!mRecompilationRequired && mProgram.compiled())
    { return; }

    treeutil::checkGLError("Before ShaderProgram compilation");
    mProgram.compile(mTransformFeedbackOutputs);
    treeutil::checkGLError("After ShaderProgram compilation");

    mRecompilationRequired = false;
}

ShaderProgramHelper ShaderProgramFactory::renderProgram()
{ return { }; }

UniformSet::UniformSet()
{ /* Automatic */ }

std::size_t UniformSet::setUniforms(const ShaderProgram &program) const
{
    auto uniformsSet{ 0u };

    for (const auto &uniformMapRec : mUniformMap)
    { uniformsSet += uniformMapRec.second->setUniform(program); }

    return uniformsSet;
}

std::vector<std::string> UniformSet::getUniformIdentifiers() const
{
    std::vector<std::string> result{ };
    result.reserve(mUniformMap.size());

    for (const auto &uniformMapRec : mUniformMap)
    { result.emplace_back(uniformMapRec.first); }

    return result;
}

void UniformSet::describe(std::ostream &out, const std::string &indent) const
{
    out << "[ UniformSet: \n"
        << indent << "\tUniform Count = " << mUniformMap.size() << "\n"
        << indent << "\tUniforms = " << (mUniformMap.empty() ? "NONE\n" : "\n");
    auto counter{ 0u };
    for (const auto &uniformRec : mUniformMap)
    {
        out << indent << "\t\t" << counter++ << " : ";
        uniformRec.second->describe(out, indent + "\t\t");
        out << "\n";
    }
    out << indent << " ]";
}

} // namespace treerndr

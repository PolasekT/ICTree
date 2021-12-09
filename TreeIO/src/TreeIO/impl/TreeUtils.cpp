/**
 * @author Tomas Polasek
 * @date 11.20.2019
 * @version 1.0
 * @brief Utilities and statistics for the treeio::Tree class.
 */

#include "TreeUtils.h"

namespace treeutil 
{

bool saveTextToFile(std::string filename, std::string contents) {

    // Create output file-name:
    const auto basePath{ treeutil::filePath(filename) };
    const auto fileBaseName{ treeutil::fileBaseName(filename) };
    const auto extension{ treeutil::fileExtension(filename) };
    const auto resultFilePath{ basePath + (basePath.length()>0?sysSepStr():"") + fileBaseName + extension };

    // Save the file:
    if(basePath.length() > 0)
        std::filesystem::create_directory(basePath);

    std::ofstream outfile (resultFilePath);
    if (outfile.is_open())
    {
        outfile << contents;
        outfile.close();
        return false;
    } else
    { Error << "Exception thrown while saving file \"" << filename.c_str() << "\"" << std::endl; return true; }
}

} // namespace treeutil

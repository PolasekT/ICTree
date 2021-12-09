#################################################################################
# @project cGANTree
# @brief Tools and utility functions for CMake.
# @author Tomas Polasek
#################################################################################

cmake_minimum_required (VERSION 3.9)

# Source: https://stackoverflow.com/a/697745
# Perform configure_file function on each file in the input directory.
macro(configure_directory srcDir destDir)
    message(STATUS "Configuring directory ${destDir}")
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            message(STATUS "Configuring file ${templateFile}")
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    @ONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(configure_directory srcDir destDir)

# Set CMAKE_BINARY_DIR and CMAKE_RUNTIME/LIBRARY/ARCHIVE_OUTPUT_DIRECTORY variables.
macro(set_binary_dir name)
    if (BUILD_ARCHITECTURE STREQUAL "x86")
        set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/${name}/build/${CMAKE_BUILD_TYPE}/x86)
    elseif (BUILD_ARCHITECTURE STREQUAL "x64")
        set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/${name}/build/${CMAKE_BUILD_TYPE}/x64)
    else ()
        set (CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/${name}/build/${CMAKE_BUILD_TYPE}/x64)
    endif ()

    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/arc)
endmacro(set_binary_dir name)

# Helper function to compress shaders into one C++ include-able file. Inspired by: https://stackoverflow.com/a/27206982
function (compress_resources input_directory output_file)
    # Initialize the output file.
    string (TIMESTAMP today_date "%m.%d.%Y")
    file (WRITE ${output_file}
"/**
 * @author Tomas Polasek, Automated
 * @date ${today_date}
 * @version 1.0
 * @brief Automatically generated header file containing resource data.
 */

namespace treerndr\n{\n\nnamespace glsl\n{
"
    )
    # Get list of files to compress.
    file (GLOB input_files ${input_directory}/*)

    foreach (input_file ${input_files})
        # Get only the filename.
        string (REGEX MATCH "([^/]+)$" filename ${input_file})
        # Make the name compatible with C/C++, using CONST_NAME naming.
        string (REGEX REPLACE "\\.| |-" "_" filename ${filename})
        string (TOUPPER ${filename} filename)

        # Read text data from file.
        file (READ ${input_file} text_data)

        # Process all includes recursively until there are no more found.
        set (found_include TRUE)
        set (included_files "")
        while (${found_include})
            # Reset progress.
            set (found_include FALSE)
            # Find all includes.
            string (REGEX MATCHALL "\\#include \"([^\"]+)\"" include_directives ${text_data})

            # Process includes, replacing them with the content of the target file.
            foreach (include_directive ${include_directives})
                # Extract the filename of the included file.
                string (REGEX REPLACE "\\#include \"([^\"]+)\"" "\\1" included_file ${include_directive})
                # Only include each file once.
                list (FIND included_files "${included_file}" included_idx)
                if (${included_idx} EQUAL -1)
                    # Read the file content.
                    file (READ ${input_directory}/${included_file} included_text)
                    # Remember we included this file for next time.
                    list (APPEND included_files "${included_file}")
                else ()
                    # Already included, just replace the include with empty string.
                    set (included_text "")
                endif ()
                # Replace the include directive with the text.
                string (REPLACE "${include_directive}" "${included_text}" text_data "${text_data}")
                # We found an include, continue recursively.
                set (found_include TRUE)
            endforeach ()
        endwhile ()

        # Conversion to HEX is available only from CMake 3.18 -> save and reload as hex...
        file (WRITE ${output_file}.tmp "${text_data}")
        file (READ ${output_file}.tmp file_data HEX)
        # Convert hexadecimal data to C/C++ byte format.
        string (REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," file_data ${file_data})

        # Append the new variables to the output file.
        file (APPEND ${output_file}
"
static constexpr char ${filename}[]{ ${file_data}0x00 };\n
static constexpr unsigned long ${filename}_size{ sizeof(${filename}) - 1 };\n
"
        )
    endforeach ()

    # Finalize the file content
    file (APPEND ${output_file} "} // namespace glsl\n\n} // namespace treerndr\n")
    # Clean up the temporary files.
    if (EXISTS ${output_file}.tmp)
        file (REMOVE ${output_file}.tmp)
    endif ()
endfunction ()

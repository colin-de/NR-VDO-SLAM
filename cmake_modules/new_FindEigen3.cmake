# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: alexs.mac@gmail.com (Alex Stewart)
#
# Changes 2020 by nikolaus@nikolaus-demmel.de (Nikolaus Demmel) 
# distributed under the same BSD 3-clause license:
# - Rename to FindEigen3.cmake and make compatible with the cmake
#   config exported by Eigen since version 3.3.3, i.e. we export
#   target Eigen3::Eigen.

# FindEigen3.cmake - Find Eigen library, version >= 3.
#
# This module defines the following target with exported include directory:
#
# Eigen3::Eigen 
#
# This module defines the following variables (preferred use is the target):
#
# EIGEN3_FOUND: TRUE iff Eigen is found.
# EIGEN3_INCLUDE_DIRS: Include directories for Eigen.
# EIGEN3_VERSION: Extracted from Eigen/src/Core/util/Macros.h
# EIGEN3_VERSION_MAJOR: Equal to 3 if EIGEN_VERSION = 3.2.0
# EIGEN3_VERSION_MINOR: Equal to 2 if EIGEN_VERSION = 3.2.0
# EIGEN3_VERSION_PATCH: Equal to 0 if EIGEN_VERSION = 3.2.0
# FOUND_INSTALLED_EIGEN_CMAKE_CONFIGURATION: True iff the version of Eigen
#                                            found was built & installed /
#                                            exported as a CMake package.
#
# The following variables control the behaviour of this module:
#
# EIGEN3_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION: TRUE/FALSE, iff TRUE then
#                           then prefer using an exported CMake configuration
#                           generated by Eigen over searching for the
#                           Eigen components manually.  Otherwise (FALSE)
#                           ignore any exported Eigen CMake configurations and
#                           always perform a manual search for the components.
#                           Default: TRUE iff user does not define this variable
#                           before we are called, and does NOT specify
#                           EIGEN3_INCLUDE_DIR_HINTS, otherwise FALSE.
# EIGEN3_INCLUDE_DIR_HINTS: List of additional directories in which to
#                           search for eigen includes, e.g: /timbuktu/eigen3.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# EIGEN3_INCLUDE_DIR: Include directory for Eigen, not including the
#                     include directory of any dependencies.

# Called if we failed to find Eigen or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(EIGEN3_REPORT_NOT_FOUND REASON_MSG)
  unset(EIGEN3_FOUND)
  unset(EIGEN3_INCLUDE_DIRS)
  unset(FOUND_INSTALLED_EIGEN_CMAKE_CONFIGURATION)
  # Make results of search visible in the CMake GUI if Eigen has not
  # been found so that user does not have to toggle to advanced view.
  mark_as_advanced(CLEAR EIGEN3_INCLUDE_DIR)
  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  if (Eigen3_FIND_QUIETLY)
    message(STATUS "Failed to find Eigen - " ${REASON_MSG} ${ARGN})
  elseif (Eigen3_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find Eigen - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    message("-- Failed to find Eigen - " ${REASON_MSG} ${ARGN})
  endif ()
  return()
endmacro(EIGEN3_REPORT_NOT_FOUND)

# Protect against any alternative find_package scripts for this library having
# been called previously (in a client project) which set EIGEN3_FOUND, but not
# the other variables we require / set here which could cause the search logic
# here to fail.
unset(EIGEN3_FOUND)

# -----------------------------------------------------------------
# By default, if the user has expressed no preference for using an exported
# Eigen CMake configuration over performing a search for the installed
# components, and has not specified any hints for the search locations, then
# prefer an exported configuration if available.
if (NOT DEFINED EIGEN3_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION
    AND NOT EIGEN3_INCLUDE_DIR_HINTS)
  message(STATUS "No preference for use of exported Eigen CMake configuration "
    "set, and no hints for include directory provided. "
    "Defaulting to preferring an installed/exported Eigen CMake configuration "
    "if available.")
  set(EIGEN3_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION TRUE)
endif()


message(STATUS "Running FindEigen3.cmake")


if (EIGEN3_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION)
  # Try to find an exported CMake configuration for Eigen.
  #
  # We search twice, s/t we can invert the ordering of precedence used by
  # find_package() for exported package build directories, and installed
  # packages (found via CMAKE_SYSTEM_PREFIX_PATH), listed as items 6) and 7)
  # respectively in [1].
  #
  # By default, exported build directories are (in theory) detected first, and
  # this is usually the case on Windows.  However, on OS X & Linux, the install
  # path (/usr/local) is typically present in the PATH environment variable
  # which is checked in item 4) in [1] (i.e. before both of the above, unless
  # NO_SYSTEM_ENVIRONMENT_PATH is passed).  As such on those OSs installed
  # packages are usually detected in preference to exported package build
  # directories.
  #
  # To ensure a more consistent response across all OSs, and as users usually
  # want to prefer an installed version of a package over a locally built one
  # where both exist (esp. as the exported build directory might be removed
  # after installation), we first search with NO_CMAKE_PACKAGE_REGISTRY which
  # means any build directories exported by the user are ignored, and thus
  # installed directories are preferred.  If this fails to find the package
  # we then research again, but without NO_CMAKE_PACKAGE_REGISTRY, so any
  # exported build directories will now be detected.
  #
  # To prevent confusion on Windows, we also pass NO_CMAKE_BUILDS_PATH (which
  # is item 5) in [1]), to not preferentially use projects that were built
  # recently with the CMake GUI to ensure that we always prefer an installed
  # version if available.
  #
  # [1] http://www.cmake.org/cmake/help/v2.8.11/cmake.html#command:find_package
  find_package(Eigen3 QUIET
                      NO_MODULE
                      NO_CMAKE_PACKAGE_REGISTRY
                      NO_CMAKE_BUILDS_PATH)
  if (EIGEN3_FOUND)
    message(STATUS "Found installed version of Eigen: ${Eigen3_DIR}")
  else()
    # Failed to find an installed version of Eigen, repeat search allowing
    # exported build directories.
    message(STATUS "Failed to find installed Eigen CMake configuration, "
      "searching for Eigen build directories exported with CMake.")
    # Again pass NO_CMAKE_BUILDS_PATH, as we know that Eigen is exported and
    # do not want to treat projects built with the CMake GUI preferentially.
    find_package(Eigen3 QUIET
                        NO_MODULE
                        NO_CMAKE_BUILDS_PATH)
    if (EIGEN3_FOUND)
      message(STATUS "Found exported Eigen build directory: ${Eigen3_DIR}")
    endif()
  endif()
  if (EIGEN3_FOUND)
    set(FOUND_INSTALLED_EIGEN_CMAKE_CONFIGURATION TRUE)
    if (TARGET Eigen3::Eigen)
      message(STATUS "Relying on Eigen's cmake config exported Eigen3::Eigen "
        "target and defined variables.")
    else()      
      eigen3_report_not_found(
        "Found eigen via cmake config, but doesn't define target Eigen3::Eigen")
    endif()
    return()
  else()
    message(STATUS "Failed to find an installed/exported CMake configuration "
      "for Eigen, will perform search for installed Eigen components.")
  endif()
endif()

if (NOT EIGEN3_FOUND)
  # Search user-installed locations first, so that we prefer user installs
  # to system installs where both exist.
  list(APPEND EIGEN3_CHECK_INCLUDE_DIRS
    /usr/local/include
    /usr/local/homebrew/include # Mac OS X
    /opt/local/var/macports/software # Mac OS X.
    /opt/local/include
    /usr/include)
  # Additional suffixes to try appending to each search path.
  list(APPEND EIGEN3_CHECK_PATH_SUFFIXES
    eigen3 # Default root directory for Eigen.
    Eigen/include/eigen3 # Windows (for C:/Program Files prefix) < 3.3
    Eigen3/include/eigen3 ) # Windows (for C:/Program Files prefix) >= 3.3

  # Search supplied hint directories first if supplied.
  find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    HINTS ${EIGEN3_INCLUDE_DIR_HINTS}
    PATHS ${EIGEN3_CHECK_INCLUDE_DIRS}
    PATH_SUFFIXES ${EIGEN3_CHECK_PATH_SUFFIXES})

  if (NOT EIGEN3_INCLUDE_DIR OR
      NOT EXISTS ${EIGEN3_INCLUDE_DIR})
    eigen3_report_not_found(
      "Could not find eigen3 include directory, set EIGEN3_INCLUDE_DIR to "
      "path to eigen3 include directory, e.g. /usr/local/include/eigen3.")
  endif ()

  # Mark internally as found, then verify. EIGEN3_REPORT_NOT_FOUND() unsets
  # if called.
  set(EIGEN3_FOUND TRUE)
endif()

# Extract Eigen version from Eigen/src/Core/util/Macros.h
if (EIGEN3_INCLUDE_DIR)
  set(EIGEN3_VERSION_FILE ${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h)
  if (NOT EXISTS ${EIGEN3_VERSION_FILE})
    eigen3_report_not_found(
      "Could not find file: ${EIGEN3_VERSION_FILE} "
      "containing version information in Eigen install located at: "
      "${EIGEN3_INCLUDE_DIR}.")
  else ()
    file(READ ${EIGEN3_VERSION_FILE} EIGEN3_VERSION_FILE_CONTENTS)

    string(REGEX MATCH "#define EIGEN_WORLD_VERSION [0-9]+"
      EIGEN3_VERSION_MAJOR "${EIGEN3_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define EIGEN_WORLD_VERSION ([0-9]+)" "\\1"
      EIGEN3_VERSION_MAJOR "${EIGEN3_VERSION_MAJOR}")

    string(REGEX MATCH "#define EIGEN_MAJOR_VERSION [0-9]+"
      EIGEN3_VERSION_MINOR "${EIGEN3_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define EIGEN_MAJOR_VERSION ([0-9]+)" "\\1"
      EIGEN3_VERSION_MINOR "${EIGEN3_VERSION_MINOR}")

    string(REGEX MATCH "#define EIGEN_MINOR_VERSION [0-9]+"
      EIGEN3_VERSION_PATCH "${EIGEN3_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define EIGEN_MINOR_VERSION ([0-9]+)" "\\1"
      EIGEN3_VERSION_PATCH "${EIGEN3_VERSION_PATCH}")

    # This is on a single line s/t CMake does not interpret it as a list of
    # elements and insert ';' separators which would result in 3.;2.;0 nonsense.
    set(EIGEN3_VERSION_STRING "${EIGEN3_VERSION_MAJOR}.${EIGEN3_VERSION_MINOR}.${EIGEN3_VERSION_PATCH}")
    set(EIGEN3_VERSION "${EIGEN3_VERSION_STRING}")
  endif ()
endif ()

# Set standard CMake FindPackage variables if found.
if (EIGEN3_FOUND)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif ()

# Export target if found
if (EIGEN3_FOUND)
  # Create imported target Eigen3::Eigen
  add_library(Eigen3::Eigen INTERFACE IMPORTED)

  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
  )
endif ()


# Handle REQUIRED / QUIET optional arguments and version.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
  REQUIRED_VARS EIGEN3_INCLUDE_DIRS
  VERSION_VAR EIGEN3_VERSION)


# Only mark internal variables as advanced if we found Eigen, otherwise
# leave it visible in the standard GUI for the user to set manually.
if (EIGEN3_FOUND)
  mark_as_advanced(FORCE EIGEN3_INCLUDE_DIR
    Eigen3_DIR) # Autogenerated by find_package(Eigen3)
endif ()
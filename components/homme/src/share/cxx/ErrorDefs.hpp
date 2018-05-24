/*********************************************************************************
 *
 * Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * For five (5) years from  the United States Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in this data to reproduce, prepare derivative works, and perform
 * publicly and display publicly, by or on behalf of the Government. There is
 * provision for the possible extension of the term of this license. Subsequent
 * to that period or any extension granted, the United States Government is
 * granted for itself and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable worldwide license in this data to reproduce, prepare derivative
 * works, distribute copies to the public, perform publicly and display publicly,
 * and to permit others to do so. The specific term of the license can be
 * identified by inquiry made to National Technology and Engineering Solutions of
 * Sandia, LLC or DOE.
 *
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
 * ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
 * ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
 * LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
 * USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 *
 * Any licensee of this software has the obligation and responsibility to abide
 * by the applicable export control laws, regulations, and general prohibitions
 * relating to the export of technical data. Failure to obtain an export control
 * license or other authority from the Government may result in criminal
 * liability under U.S. laws.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 *     - Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *     - Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the documentation
 *       and/or other materials provided with the distribution.
 *     - Neither the name of Sandia Corporation,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this Software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************/

#ifndef HOMMEXX_ERRORDEFS_HPP
#define HOMMEXX_ERRORDEFS_HPP

#ifndef NDEBUG
#define DEBUG_PRINT(...) \
  do {                   \
    printf(__VA_ARGS__); \
  } while(false)
// This macro always evaluates eval, but
// This enables us to define variables specifically for use
// in asserts Note this can still cause issues
#define DEBUG_EXPECT(eval, expected) \
do {                                 \
  auto v = eval;                     \
  assert(v == expected);             \
} while(false)
#else
#define DEBUG_PRINT(...) \
do {                     \
} while(false)
#define DEBUG_EXPECT(eval, expected) \
do {                                 \
  eval;                              \
} while(false)
#endif

#ifdef DEBUG_TRACE
#define TRACE_PRINT(...) \
do {                     \
  printf(__VA_ARGS__);   \
} while(false)
#else
#define TRACE_PRINT(...) \
do {                     \
} while(false)
#endif

#include <HommexxEnums.hpp>
#include <string>
#include <sstream>

namespace Homme {
namespace Errors {

void runtime_check(bool cond, const std::string& message, int code);
void runtime_abort(const std::string& message, int code);

static constexpr int err_unknown_option               = 11;
static constexpr int err_not_implemented              = 12;
static constexpr int err_invalid_options_combination  = 13;
static constexpr int err_negative_layer_thickness     = 101;

template<typename T>
void option_error(const std::string& location,
                  const std::string& option,
                  const T& value)
{
  std::stringstream msg;
  msg << "Error in " << location << ": " << "unsupported value '"
      << value << "' for input parameter '" << option << "'.";

  runtime_abort(msg.str(),err_not_implemented);
}

template<typename T>
void check_option(const std::string& location,
                  const std::string& option,
                  const T& actual_value,
                  const std::initializer_list<T>& admissible_values)
{
  bool bad_value = true;
  for (const auto& value : admissible_values) {
    if (value==actual_value) {
      bad_value = false;
    }
  }

  if (bad_value) {
    option_error(location,option,actual_value);
  }
}

template<typename T>
void check_option (const std::string& location,
                   const std::string& option,
                   const T& value, const T& ref_value,
                   const ComparisonOp& relation)
{
  bool bad_inputs = false;
  switch (relation) {
    case ComparisonOp::EQ: if (value!=ref_value) { bad_inputs = true; } break;
    case ComparisonOp::NE: if (value==ref_value) { bad_inputs = true; } break;
    case ComparisonOp::GT: if (value<=ref_value) { bad_inputs = true; } break;
    case ComparisonOp::LT: if (value>=ref_value) { bad_inputs = true; } break;
    case ComparisonOp::GE: if (value<ref_value)  { bad_inputs = true; } break;
    case ComparisonOp::LE: if (value>ref_value)  { bad_inputs = true; } break;
  }

  if (bad_inputs) {
    const std::string cmp_str[6] {"==", "!=", ">", "<", ">=", "<="};
    std::stringstream msg;
    msg << "Error in " << location << ": " << "unsupported value '"
        << value << "' for input parameter '" << option << "'.\n"
        << " The value should satisfy " << option << " " << cmp_str[etoi(relation)]
        << " " << ref_value << ".";

    runtime_abort(msg.str(),err_invalid_options_combination);
  }
}

template<typename T1, typename T2>
void check_options_relation(const std::string& location,
                            const std::string& option1,
                            const std::string& option2,
                            const T1& value1, const T2& value2,
                            const ComparisonOp& relation)
{
  bool bad_inputs = false;
  switch (relation) {
    case ComparisonOp::EQ: if (value1!=value2) { bad_inputs = true; } break;
    case ComparisonOp::NE: if (value1==value2) { bad_inputs = true; } break;
    case ComparisonOp::GT: if (value1<=value2) { bad_inputs = true; } break;
    case ComparisonOp::LT: if (value1>=value2) { bad_inputs = true; } break;
    case ComparisonOp::GE: if (value1<value2)  { bad_inputs = true; } break;
    case ComparisonOp::LE: if (value1>value2)  { bad_inputs = true; } break;
  }

  if (bad_inputs) {
    const std::string cmp_str[6] {"==", "!=", ">", "<", ">=", "<="};
    std::stringstream msg;
    msg << "Error in " << location << ": " << "unsupported combination for input parameters '"
        << option1 << "' (" << value1 << ") and '" << option2 << "' (" << value2 << ").\n"
        << " The two should satisfy " << option1 << " " << cmp_str[etoi(relation)]
        << " " << option2 << ".";

    runtime_abort(msg.str(),err_invalid_options_combination);
  }
}

} // namespace Errors
} // namespace Homme

#endif // HOMMEXX_ERRORDEFS_HPP

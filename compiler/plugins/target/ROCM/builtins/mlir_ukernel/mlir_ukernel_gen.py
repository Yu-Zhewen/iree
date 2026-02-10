#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Template processor for MLIR microkernel code generation.

This script processes .mlir.in template files and generates .mlir output files
using Python-like syntax.

Usage:
    python mlir_ukernel_gen.py template.mlir.in -o output.mlir \\
        -D INTRINSICS_M=4 INTRINSICS_N=8 ARCH=gfx942

Template syntax:
    ${VAR}              - Variable substitution
    ${VAR * 2}          - Expression evaluation
    $for i in range(4): - Loop unrolling
    $if condition:      - Conditional generation
    $else:              - Alternative branch
"""

import argparse
import io
import re
from typing import Dict, Any

ELEM_TYPE_BITS = {
    "bf16": 16,
    "f16": 16,
    "f32": 32,
    "f4E2M1FN": 4,
    "f8E4M3FN": 8,
    "f8E4M3FNUZ": 8,
}


def fold1(val):
    """
    Fold dimension if value is 1.
    Returns 'valx' if val != 1, else empty string.
    """
    return f"{val}x" if val != 1 else ""


def expand_reassoc_fold1(brackets):
    """
    Build tensor.expand_shape reassociation from a list of brackets.

    Each bracket is a list describing the expanded dimensions that map to one
    base dimension. The number of expand dims for a bracket is len(bracket).
    If a bracket has exactly one element and that element equals 1 (and it is
    not the first bracket), that bracket is merged with the next.

    Example (in template: ${EXPAND_REASSOC_FOLD1(...)}):
      EXPAND_REASSOC_FOLD1([[1], ["?", 4], [SUBGROUPS_M], [INTRINSICS_M], [4],
                            [8, 2], [INTRINSICS_K], [INTERNAL_K]])
      With SUBGROUPS_M=1, INTRINSICS_M=8, the 3rd bracket merges with the 4th,
      giving 7 groups instead of 8:
      [[0], [1, 2], [3, 4], [5], [6, 7], [8], [9]].
    """
    # Assign consecutive expand indices to each bracket.
    expand_indices_per_bracket = []
    idx = 0
    for b in brackets:
        size = len(b)
        expand_indices_per_bracket.append(list(range(idx, idx + size)))
        idx += size

    # Merge: when a bracket has a single element and that value is 1, merge
    # with next (except for the first bracket, which is never merged).
    merged_groups = []
    i = 0
    while i < len(expand_indices_per_bracket):
        group = list(expand_indices_per_bracket[i])
        if (
            i > 0
            and i < len(brackets)
            and len(brackets[i]) == 1
            and brackets[i][0] == 1
            and i + 1 < len(expand_indices_per_bracket)
        ):
            group.extend(expand_indices_per_bracket[i + 1])
            i += 1
        merged_groups.append(group)
        i += 1

    # Format as reassociation attribute.
    return (
        "["
        + ", ".join("[" + ", ".join(str(x) for x in g) + "]" for g in merged_groups)
        + "]"
    )


def size_constraints(params: Dict[str, Any]) -> str:
    """
    Build the iteration_sizes_constraints list for rocm.ukernel_info.

    Returns a multi-line string of #rocm.ukernel_interation_size_constraint<...>
    for indices 0, 1, 2, formatted with one constraint per block (same style as
    hand-written ukernels). For each index i, a constraint is emitted only if at least
    one of SIZE_MIN_i, SIZE_MAX_i, SIZE_DIV_i is present in params (i.e. was passed
    via -D). Only fields that were provided are included in each constraint.

    Use in template as (template line has 8 spaces before ${SIZE_CONSTRAINTS}):
      iteration_sizes_constraints = [
        ${SIZE_CONSTRAINTS}
      ]
    """
    # Template has 8 spaces before ${SIZE_CONSTRAINTS}; only the first line gets that
    # prefix, so we add base_indent to every line we output.
    base_indent = "        "  # 8 spaces
    indent_inner = "  "  # index/size_* at column 10

    def _constraint_for(i: int) -> str:
        key_min = f"SIZE_MIN_{i}"
        key_max = f"SIZE_MAX_{i}"
        key_div = f"SIZE_DIV_{i}"
        if key_min not in params and key_max not in params and key_div not in params:
            return ""
        lines = ["#rocm.ukernel_interation_size_constraint<"]
        inner = [f"index = {i}"]
        if key_min in params:
            inner.append(f"size_min = {params[key_min]}")
        if key_max in params:
            inner.append(f"size_max = {params[key_max]}")
        if key_div in params:
            inner.append(f"size_div = {params[key_div]}")
        # Format inner params: comma after all but last
        for j, part in enumerate(inner):
            suffix = "," if j < len(inner) - 1 else ""
            lines.append(f"{indent_inner}{part}{suffix}")
        lines.append(">")
        return "\n".join(lines)

    constraints = []
    for i in [0, 1, 2]:
        s = _constraint_for(i)
        if s:
            constraints.append(s)
    raw = ",\n".join(constraints)
    # Prepend base_indent to every line so indentation is correct when template has 8 spaces.
    return "\n".join(base_indent + line for line in raw.split("\n"))


def extract_internal_k(intrinsic_name: str) -> int:
    """
    Extract the K dimension from an intrinsic name and divide by 4.

    This aligns with:
    https://github.com/iree-org/iree/blob/de381bdda5bbcfdd3e0897b2c30b9e781c4d3846/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp#L230-L239

    Examples:
        MFMA_F32_16x16x16_F16 -> 16 // 4 = 4
        MFMA_F32_16x16x32_F8E4M3FNUZ -> 32 // 4 = 8
        WMMA_F32_16x16x16_F16 -> 16 // 4 = 4
    """
    match = re.search(r"_(\d+)x(\d+)x(\d+)_", intrinsic_name)
    if not match:
        raise ValueError(
            f"Could not extract K dimension from intrinsic: {intrinsic_name}"
        )

    m, n, k = int(match.group(1)), int(match.group(2)), int(match.group(3))

    # Assert that M and N are 16 as expected for supported intrinsics.
    assert m == 16, f"Expected M=16 in intrinsic {intrinsic_name}, got M={m}"
    assert n == 16, f"Expected N=16 in intrinsic {intrinsic_name}, got N={n}"

    return k // 4


def process_template(text: str, params: Dict[str, Any]) -> str:
    """Process template with Python-like control flow.

    Lines starting with $ (but not ${) are treated as Python statements.
    Other lines are output directly with ${expr} evaluated and substituted.

    Example (INTRINSICS_M=4):
        $for i in range(INTRINSICS_M):
          %val_${i} = arith.constant ${i} : index

        Generates:
          %val_0 = arith.constant 0 : index
          %val_1 = arith.constant 1 : index
          %val_2 = arith.constant 2 : index
          %val_3 = arith.constant 3 : index
    """

    def extract_leading_whitespace(line):
        """Get leading whitespace from a line."""
        match = re.match(r"^\s*", line)
        return match.group(0) if match else ""

    def escape_line(line):
        """Convert a line with ${expr} to Python print statement parts."""
        output_parts = []
        while "${" in line:
            start_pos = line.index("${")
            end_pos = line.index("}", start_pos + 2)
            if start_pos != 0:
                output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
            expr = line[start_pos + 2 : end_pos]
            output_parts.append("str(" + expr + ")")
            line = line[end_pos + 1 :]
        if line:
            output_parts.append('"' + line.replace('"', '\\"') + '"')
        return " + ".join(output_parts) if output_parts else '""'

    # Convert template to Python code.
    input_lines = text.splitlines()
    python_lines = []
    blank_lines = 0
    last_indent = ""
    indent_stack = [("", "")]
    python_block_start = True

    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue

        input_indent = extract_leading_whitespace(input_line)

        # Adjust indentation stack.
        if python_block_start:
            assert input_indent.startswith(
                last_indent
            ), f"Indentation error: expected indent to start with '{last_indent}', got '{input_indent}'"
            extra_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_indent
            indent_stack.append((input_indent, python_indent))
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                indent_stack.pop()

        python_block_start = False
        python_indent = indent_stack[-1][1]
        stripped_line = input_line.strip()

        if stripped_line.startswith("$") and not stripped_line.startswith("${"):
            if stripped_line.endswith(":"):
                python_block_start = True

            while blank_lines > 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1

            python_lines.append(python_indent + stripped_line[1:])
        else:
            while blank_lines > 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1

            line_after_indent = input_line[len(python_indent) :]
            python_lines.append(
                python_indent
                + "print(%s, file=OUT_STREAM)" % escape_line(line_after_indent)
            )

        last_indent = input_indent

    while blank_lines > 0:
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    # Set up execution context.
    exec_globals = params.copy()
    exec_globals["FOLD1"] = fold1
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream
    exec_globals["EXPAND_REASSOC_FOLD1"] = expand_reassoc_fold1
    exec_globals["SIZE_CONSTRAINTS"] = size_constraints(params)

    # Compile and execute the generated Python code.
    python_code = "\n".join(python_lines)
    try:
        python_bytecode = compile(python_code, "<template>", "exec")
        exec(python_bytecode, exec_globals)
    except Exception as e:
        # Show the generated Python code for debugging.
        print("Generated Python code:")
        for i, line in enumerate(python_lines, 1):
            print(f"{i:4d}: {line}")
        raise ValueError(f"Failed to execute template: {e}")

    return output_stream.getvalue()


def parse_define(define_str: str) -> tuple:
    """Parse a -D VAR=VALUE string into (var, value)."""
    if "=" in define_str:
        var, value = define_str.split("=", 1)
        # Try to convert to int if possible.
        try:
            value = int(value)
        except ValueError:
            pass
        return var.strip(), value
    else:
        return define_str.strip(), True


def main():
    parser = argparse.ArgumentParser(
        description="Process MLIR microkernel templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mlir_ukernel_gen.py template.mlir.in -o output.mlir \\
        -D INTRINSICS_M=4 INTRINSICS_N=8 ARCH=gfx942
        """,
    )

    parser.add_argument("template", type=str, help="Path to the .mlir.in template file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "-D",
        "--define",
        type=str,
        nargs="+",
        default=[],
        help="Define parameters: -D VAR1=VALUE1 VAR2=VALUE2 ...",
    )

    args = parser.parse_args()

    # Build params from defines.
    params = {}
    for define in args.define:
        var, value = parse_define(define)
        params[var] = value

    # Derive ELEM_BITS from ELEM_TYPE.
    elem_type = params["ELEM_TYPE"]
    assert elem_type in ELEM_TYPE_BITS, f"Invalid element type: {elem_type}"
    params["ELEM_BITS"] = ELEM_TYPE_BITS[elem_type]

    # Derive INTERNAL_K from INTRINSIC.
    intrinsic = params["INTRINSIC"]
    params["INTERNAL_K"] = extract_internal_k(intrinsic)

    # Note: Parameter validation now happens via ${ASSERT(...)} in templates.

    # Read template.
    with open(args.template, "r") as f:
        template = f.read()

    # Process template with control flow and expressions.
    output = process_template(template, params)

    # Write output.
    with open(args.output, "w") as f:
        f.write(output)
    print(f"Generated: {args.output}")


if __name__ == "__main__":
    main()

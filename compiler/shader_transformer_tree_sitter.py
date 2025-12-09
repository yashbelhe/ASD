# %%
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import os
import copy
import subprocess
import re


def is_continuous_function(func_node, source_code_bytes):
    """Check if function is marked as continuous"""
    prev_sibling = func_node.prev_sibling
    if prev_sibling and prev_sibling.type == 'comment':
        comment = source_code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf8')
        return '[NoDiscontinuity]' in comment
    return False

def find_function_calls(node, source_code_bytes):
    """Find all function calls within a node"""
    calls = set()
    if node.type == 'call_expression':
        func_name = next((c for c in node.children if c.type == 'identifier'), None)
        if func_name:
            name = source_code_bytes[func_name.start_byte:func_name.end_byte].decode('utf8')
            calls.add(name)
    
    for child in node.children:
        calls.update(find_function_calls(child, source_code_bytes))
    
    return calls

def find_function_definitions(node, source_code_bytes):
    """Find all function definitions, check for cycles, and return topologically sorted functions"""
    # First pass: collect all functions
    functions = {}
    call_graph = {}  # function -> set of functions it calls
    
    def collect_functions(node):
        if node.type == 'function_definition':
            declarator = next((c for c in node.children if c.type == 'function_declarator'), None)
            if declarator:
                identifier = next((c for c in declarator.children if c.type == 'identifier'), None)
                if identifier:
                    func_name = source_code_bytes[identifier.start_byte:identifier.end_byte].decode('utf8')
                    functions[func_name] = node
                    # Find all function calls within this function
                    call_graph[func_name] = find_function_calls(node, source_code_bytes)
        
        for child in node.children:
            collect_functions(child)
    
    collect_functions(node)
    
    # Check for cycles using DFS
    def has_cycle(func_name, visited, stack):
        if func_name in stack:
            cycle_path = ' -> '.join(list(stack) + [func_name])
            raise ValueError(f"Cycle detected in function calls: {cycle_path}")
        
        if func_name in visited:
            return False
            
        visited.add(func_name)
        stack.add(func_name)
        
        for called_func in call_graph.get(func_name, set()):
            if called_func in functions and has_cycle(called_func, visited, stack):
                return True
                
        stack.remove(func_name)
        return False
    
    # Start cycle detection from each function
    visited = set()
    for func_name in functions:
        if func_name not in visited:
            has_cycle(func_name, visited, set())
    
    # Perform topological sort
    sorted_functions = []
    visited = set()
    
    def topological_sort(func_name):
        if func_name in visited:
            return
        visited.add(func_name)
        
        # First visit all functions this one calls
        for called_func in call_graph.get(func_name, set()):
            if called_func in functions:
                topological_sort(called_func)
        
        # After visiting all dependencies, add this function
        sorted_functions.append((func_name, functions[func_name]))
    
    # Start topological sort from each function
    visited.clear()
    for func_name in functions:
        if func_name not in visited:
            topological_sort(func_name)
    
    # Return list of (name, node) tuples in topological order
    return sorted_functions

def preprocess_source(source_code, keep_disc_attributes=False):
    """Add // before any line that starts with [ (ignoring whitespace).
    If keep_disc_attributes is True, do not comment out [Disc] so it remains in the output.
    """
    lines = source_code.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('['):
            if keep_disc_attributes and stripped.startswith('[Disc]'):
                processed_lines.append(line)
                continue
            indent = line[:len(line) - len(stripped)]
            processed_lines.append(f"{indent}//{stripped}")
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def post_process_source(source_code):
    """Fix whitespace and macro placeholders."""
    def replace_hash_idx(match):
        indent, var = match.groups()
        return (
            f"{indent}if(ret_const) {{\n"
            f"{indent}    out_idx[thread_idx] = hash(out_idx[thread_idx], {var});\n"
            f"{indent}}}"
        )

    def replace_skip_hash(match):
        indent, expr = match.groups()
        expr = expr.rstrip()
        return (
            f"{indent}if(!ret_const) {{\n"
            f"{indent}    {expr};\n"
            f"{indent}}}"
        )

    source_code = re.sub(r"(?m)^([ \t]*)hash_idx\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$", replace_hash_idx, source_code)
    source_code = re.sub(r"(?m)^([ \t]*)skip_hash\s+(.*?);\s*$", replace_skip_hash, source_code)
    return source_code

def get_loop_variable(node, source_code_bytes):
    """Extract loop variable name from for statement"""
    decl = next((c for c in node.children if c.type == 'declaration'), None)
    if decl:
        init_declarator = next((c for c in decl.children if c.type == 'init_declarator'), None)
        if init_declarator:
            identifier = next((c for c in init_declarator.children if c.type == 'identifier'), None)
            if identifier:
                return source_code_bytes[identifier.start_byte:identifier.end_byte].decode('utf8')
    return None

def find_float_declaration(node, var_name, source_code_bytes):
    """Find if a variable was declared as float by looking at previous siblings"""
    current = node
    while current:
        # Look at all previous siblings
        sibling = current.prev_sibling
        while sibling:
            if sibling.type == 'declaration':
                type_node = next((c for c in sibling.children if c.type == 'primitive_type'), None)
                if type_node and source_code_bytes[type_node.start_byte:type_node.end_byte].decode('utf8') == 'float':
                    init_decl = next((c for c in sibling.children if c.type == 'init_declarator'), None)
                    if init_decl:
                        var_node = next((c for c in init_decl.children if c.type == 'identifier'), None)
                        if var_node and source_code_bytes[var_node.start_byte:var_node.end_byte].decode('utf8') == var_name:
                            return True
            sibling = sibling.prev_sibling
        current = current.parent
    return False

def _extract_disc_hash(node, source_code_bytes, window=256):
    start = max(0, node.start_byte - window)
    segment = source_code_bytes[start:node.start_byte].decode('utf8')
    m = re.search(r'\[Hash\s+([A-Za-z_][A-Za-z0-9_]*)\]', segment)
    return m.group(1) if m else None


def is_disc_if(node, source_code_bytes):
    """Check if condition has a [Disc] marker comment."""
    if node.type != 'if_statement':
        return False
    
    prev_sibling = node.prev_sibling
    if prev_sibling and prev_sibling.type == 'comment':
        comment_text = source_code_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf8')
        if '[Disc]' in comment_text:
            return True
    return False
    
def is_disc_else(node, source_bytes):
    """Check if node is a discontinuous else statement"""
    if node.type != 'else_clause':
        return False
    parent = node.parent
    if parent and parent.type == 'if_statement':
        return is_disc_if(parent, source_bytes)
    return False

def get_offset_from_cache(node, info, offset_type):
    """Get total offset for a node by walking up to find containing loops"""
    current = node
    offset_parts = []
    
    while current:
        node_info = info['node_info'].get(current.id)
        if node_info:
            if current.type == 'for_statement':
                # Add loop variable term
                count = node_info[f'total_{offset_type}s']
                if count > 0:
                    loop_var = node_info['loop_var']
                    offset_parts.append(f"{loop_var}*{count}")
        current = current.parent
        
    return " + ".join(offset_parts) if offset_parts else "0"

def is_for_loop(node):
    """Check if node is a for loop"""
    return node.type == 'for_statement'

def get_max_iters(node, source_bytes):
    """Get max iters for a node by looking at its immediate previous sibling"""
    prev_sibling = node.prev_sibling
    if prev_sibling and prev_sibling.type == 'comment':
        comment_text = source_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf8')
        if '[MaxIters(' in comment_text:
            start_idx = comment_text.find('(') + 1
            end_idx = comment_text.find(')')
            if start_idx > 0 and end_idx > start_idx:
                return int(comment_text[start_idx:end_idx])
    return -1

def modify_if_statement(node, source_bytes, info, modifications, if_off, hash_guard=None):
    """Modify if statement to add implicit function test"""
    # The overall structure of the if statement is:
    # if_statement
    #   if
    #   condition_clause
    #       (
    #       binary_expression
    #       )
    #   compound_statement
    #       {
    #           body
    #       }
    prev_sibling = node.prev_sibling
    if prev_sibling and prev_sibling.type == 'comment':
        # comment_text = source_bytes[prev_sibling.start_byte:prev_sibling.end_byte].decode('utf8')
        modifications.append((prev_sibling.start_byte, prev_sibling.end_byte, ""))

    binary_expr = None
    for c in node.children:
        if c.type == 'condition_clause':
            for ch in c.children:
                if ch.type == 'binary_expression':
                    binary_expr = ch
                    break
    
    if binary_expr:
        left = binary_expr.children[0]
        cond_value = source_bytes[left.start_byte:left.end_byte].decode('utf8')
        
        indent = ' ' * node.start_point[1]
        implicit_test = (
            f"\n{indent}// ---- Start: implicit function test ----\n"
            f"{indent}if (ret_impl && abs({cond_value}) < abs(impl_fn[thread_idx])) {{\n"
            f"{indent}    impl_idx[thread_idx] = {if_off};\n"
            f"{indent}    impl_fn[thread_idx] = {cond_value};\n"
            f"{indent}}}\n"
            f"{indent}// ---- End: implicit function test ----\n\n{indent}"
        )
        new_condition = f"should_force && load(impl_idx, thread_idx) == ({if_off}) ? force_value : {cond_value} > 0.0"
        
        modifications.append((node.start_byte, node.start_byte, implicit_test))
        modifications.append((binary_expr.start_byte, binary_expr.end_byte, new_condition))

    for c in node.children:
        if c.type == 'compound_statement':
            hash_expr = hash_guard if hash_guard else f"({if_off})*2"
            insert_pos = c.start_byte + 1
            if_indent = ' ' * (c.start_point[1] + 4)
            inner_indent = if_indent + ' ' * 4
            snippet = (
                f"\n{if_indent}if(ret_const) {{\n"
                f"{inner_indent}out_idx[thread_idx] = hash(out_idx[thread_idx], {hash_expr});\n"
                f"{if_indent}}}\n"
            )
            modifications.append((insert_pos, insert_pos, snippet))


def modify_else_statement(node, source_bytes, info, modifications, if_off, hash_guard=None):
    """Modify else statement to add implicit function test"""
    prev_sibling = node.prev_sibling
    if prev_sibling and prev_sibling.type == 'comment':
        modifications.append((prev_sibling.start_byte, prev_sibling.end_byte, ""))
    # NOTE: DISCONTINUOUS ELSE IF NOT SUPPORTED only if/else
    for c in node.children:
        if c.type == 'compound_statement':
            hash_expr = hash_guard if hash_guard else f"({if_off})*2 + 1"
            insert_pos = c.start_byte + 1
            if_indent = ' ' * (c.start_point[1] + 4)
            inner_indent = if_indent + ' ' * 4
            snippet = (
                f"\n{if_indent}if(ret_const) {{\n"
                f"{inner_indent}out_idx[thread_idx] = hash(out_idx[thread_idx], {hash_expr});\n"
                f"{if_indent}}}\n"
            )
            modifications.append((insert_pos, insert_pos, snippet))




def modify_out_statement(node, source_bytes, info, modifications, out_off):
    """Modify out statement to add implicit function test"""
    # right = node.children[2]
    # original_value = source_bytes[right.start_byte:right.end_byte].decode('utf8')
    # indent = len(source_bytes[node.start_byte:right.start_byte].decode('utf8').expandtabs()) - len(source_bytes[node.start_byte:right.start_byte].decode('utf8').expandtabs().lstrip())
    # Count leading whitespace on the line containing the node
    line_start = node.start_byte
    while line_start > 0 and source_bytes[line_start-1:line_start].decode('utf8') != '\n':
        line_start -= 1
    indent = len(source_bytes[line_start:node.start_byte].decode('utf8').expandtabs())
    modifications.append((node.start_byte, node.start_byte,
        f"out_idx[thread_idx] = hash(out_idx[thread_idx], {out_off});\n{' '*(indent)}"))
    # f"if(ret_const) {{ \n{' '*(indent+4)}out_idx[thread_idx] = hash(out_idx[thread_idx], {out_off});\n{' '*(indent)}}} else {{\n{' '*(indent+4)}"))
    # modifications.append((node.end_byte+1, node.end_byte+1,
    #     f"\n{' '*(indent)}}}\n"))
    # modifications.append((right.start_byte, right.end_byte, f"ret_const ? {out_off} : {original_value}"))

def modify_hash_idx_statement(node, source_bytes, info, modifications, out_off):
    """Modify hash_idx statement to manually hash additional indices"""
    identifier = next((c for c in node.children if c.type == 'identifier'), None)
    if not identifier:
        return
    var_name = source_bytes[identifier.start_byte:identifier.end_byte].decode('utf8')
    line_start = node.start_byte
    while line_start > 0 and source_bytes[line_start-1:line_start].decode('utf8') != '\n':
        line_start -= 1
    indent = len(source_bytes[line_start:node.start_byte].decode('utf8').expandtabs())
    indent_str = ' ' * indent
    replacement = (
        f"{indent_str}if(ret_const) {{\n"
        f"{indent_str}    out_idx[thread_idx] = hash(out_idx[thread_idx], {var_name});\n"
        f"{indent_str}}}\n"
    )
    modifications.append((node.start_byte, node.end_byte, replacement))

def modify_skip_hash_statement(node, source_bytes, info, modifications, out_off):
    """Modify skip_hash statement to manually hash additional indices"""
    type_identifier = next((c for c in node.children if c.type == 'type_identifier'), None)
    if not type_identifier:
        return
    body = source_bytes[type_identifier.end_byte:node.end_byte].decode('utf8').lstrip()
    line_start = node.start_byte
    while line_start > 0 and source_bytes[line_start-1:line_start].decode('utf8') != '\n':
        line_start -= 1
    indent = len(source_bytes[line_start:node.start_byte].decode('utf8').expandtabs())
    indent_str = ' ' * indent
    replacement = (
        f"{indent_str}if(!ret_const) {{\n"
        f"{indent_str}    {body.rstrip()}\n"
        f"{indent_str}}}\n"
    )
    modifications.append((node.start_byte, node.end_byte, replacement))

def modify_call_expression(node, source_bytes, info, modifications, if_off, out_off, is_start_func):
    """Modify call expression to add implicit function test"""
    arg_list = next((c for c in node.children if c.type == 'argument_list'), None)
    if arg_list:
        if is_start_func:
            new_args = f", {if_off}, {out_off}, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value"
        else:
            new_args = f", {if_off}, {out_off}, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value, thread_idx"
        modifications.append((arg_list.end_byte - 1, arg_list.end_byte - 1, new_args))

def modify_disc_func_def(node, source_bytes, modifications):
    """Modify discontinuous function definition to add extra input arguments"""
    # Get function name
    func_name = None
    for c in node.children:
        if c.type == 'function_declarator':
            for ch in c.children:
                if ch.type == 'identifier':
                    func_name = source_bytes[ch.start_byte:ch.end_byte].decode('utf8')
                    break
            if func_name:
                break
    
    
    # Special handling for run function
    if func_name == 'run':
        extra_args = [
            'DiffTensorView<float> impl_fn',
            'TensorView<int> impl_idx',
            'TensorView<int> out_idx',
            'int force_sign',
            'bool ret_const',
            'bool ret_impl',
            # 'int if_off=0',
            # 'int out_off=0',
        ]
    elif func_name == 'start':
        extra_args = [
            'int if_off',
            'int out_off',
            'bool ret_const',
            'bool ret_impl',
            'DiffTensorView<float> impl_fn',
            'TensorView<int> impl_idx',
            'TensorView<int> out_idx',
            'bool should_force',
            'bool force_value',
        ]
    else:
        extra_args = [
            'int if_off',
            'int out_off',
            'bool ret_const',
            'bool ret_impl',
            'DiffTensorView<float> impl_fn',
            'TensorView<int> impl_idx',
            'TensorView<int> out_idx',
            'bool should_force',
            'bool force_value',
            'int thread_idx'
        ]
    extra_args_str = ", ".join(extra_args)
    for c in node.children:
        if c.type == 'function_declarator':
            for ch in c.children:
                if ch.type == 'parameter_list':
                    # Add extra arguments before the closing parenthesis
                    if len(ch.children) >= 2:  # Ensure there are at least opening and closing parens
                        modifications.append((ch.children[-1].start_byte, ch.children[-1].start_byte, ", " + extra_args_str))
    
    if func_name == 'run':
        # Add force_sign check after opening brace
        for c in node.children:
            if c.type == 'compound_statement':
                modifications.append((c.start_byte + 1, c.start_byte + 1, "\n    bool should_force = force_sign != -1;\n    bool force_value = force_sign == 0;\n    int if_off = 0; \n    int out_off = 0; \n\n"))
                break
    
    

def is_output_node(node):
    if node.type != 'assignment_expression':
        return False
    left = node.children[0]
    if left.type == 'subscript_expression':
        array = left.children[0]
        return array.type == 'identifier' and array.text.decode('utf8') == 'res'
    return False

def is_hash_idx_statement(node):
    """Check if node is a hash_idx declaration statement"""
    if node.type != 'declaration':
        return False
    
    # Check if the declaration has a type specifier
    type_specifier = next((c for c in node.children if c.type == 'type_identifier'), None)
    if not type_specifier or type_specifier.text.decode('utf8') != 'hash_idx':
        return False
    return True

def extract_skip_hash_body(node, source_bytes):
    type_identifier = next((c for c in node.children if c.type == 'type_identifier'), None)
    if not type_identifier:
        return ""
    body_start = type_identifier.end_byte
    body_end = node.end_byte
    body = source_bytes[body_start:body_end].decode('utf8').lstrip()
    return body.rstrip()

def remove_line(text, start_byte, end_byte):
    text = bytearray(text)
    text[start_byte:end_byte] = b""
    return True

def is_skip_hash_statement(node):
    """Check if node is a skip_hash declaration statement"""
    if node.type != 'declaration':
        return False
    
    # Check if the declaration has a type specifier
    type_specifier = next((c for c in node.children if c.type == 'type_identifier'), None)
    if not type_specifier or type_specifier.text.decode('utf8') != 'skip_hash':
        return False
        
    return True

def is_function_definition(node):
    return node.type == 'function_definition'


def counter_list_to_string(counter_list):
    counter_list_ = counter_list.copy()
    has_int = False
    int_off = 0
    counter_list_str_ = []
    for x in counter_list_:
        if type(x) == int:
            has_int = True
            int_off += x
        elif type(x) == str:
            counter_list_str_.append(x)
        else:
            raise ValueError(f"Invalid counter list element: {x}")
    string_off = " + ".join([str(c) for c in counter_list_str_])
    int_off_str = str(int_off)

    # print(f"String off: {string_off}, int off: {int_off_str}, has_int: {has_int}")
    
    if has_int and len(counter_list_str_) > 0:
        return string_off + f" + {int_off_str}"
    elif has_int:
        return int_off_str
    elif len(counter_list_str_) > 0:
        return string_off
    else:
        assert False, "Counter list is empty"

def process_node_dfs(node, source_bytes, info, mode, mod_ctx_=None, modifications=None, parent_loop=None):
    """Single depth-first search pass to process nodes and collect modifications"""

    assert mode in ["collect", "modify"], "Invalid mode"

    # Handle preprocessor directives: don't inject/modify at the directive node itself,
    # but still traverse children so discontinuities within guarded blocks are seen.
    if node.type.startswith("preproc"):
        if mode == "collect":
            total_ifs = 0
            total_outs = 0
            for child in node.children:
                child_info = process_node_dfs(child, source_bytes, info, mode, mod_ctx_, modifications, parent_loop)
                if child_info:
                    total_ifs += child_info['total_ifs']
                    total_outs += child_info['total_outs']
            node_info = {'total_ifs': total_ifs, 'total_outs': total_outs, 'max_iters': 1}
            info['node_info'][node.id] = node_info
            return node_info
        else:  # modify mode: just recurse into children
            for child in node.children:
                process_node_dfs(child, source_bytes, info, mode, mod_ctx_, modifications, parent_loop)
            return

    if mode == "collect":
        node_info = {
            'total_ifs': 0,
            'total_outs': 0,
            'max_iters': 1
        }
        mod_ctx = None
    elif mode == "modify":
        assert mod_ctx_ is not None, "Modify context is required"
        mod_ctx = mod_ctx_
        # Copy the modify context to avoid mutating the original
        # mod_ctx = copy.deepcopy(mod_ctx_)
    
        # print(f"Node type: {node.type} \n"
        #     f"with if_off: {counter_list_to_string(mod_ctx['curr_ifs'])} \n"
        #     f"and out_off: {counter_list_to_string(mod_ctx['curr_outs'])}\n\n\n")


    # Process current node
    if is_function_definition(node):
        if mode == "modify":
            modify_disc_func_def(node, source_bytes, modifications)
    elif is_for_loop(node):
        max_iters = get_max_iters(node, source_bytes)
        if mode == "collect":
            node_info['max_iters'] = max_iters
        elif mode == "modify":
            loop_var = get_loop_variable(node, source_bytes)
            ifs_per_loop_iter = info['node_info'][node.id]['total_ifs'] // max_iters
            outs_per_loop_iter = info['node_info'][node.id]['total_outs'] // max_iters
            mod_ctx['curr_ifs'].append(f"{loop_var}*{ifs_per_loop_iter}")
            mod_ctx['curr_outs'].append(f"{loop_var}*{outs_per_loop_iter}")
    elif is_disc_if(node, source_bytes):
    # elif is_float_if(node, source_bytes):
        if mode == "collect":
            node_info['total_ifs'] += 1
        elif mode == "modify":
            hash_guard = _extract_disc_hash(node, source_bytes)
            modify_if_statement(node, source_bytes, info, modifications, counter_list_to_string(mod_ctx['curr_ifs']), hash_guard)
            mod_ctx['curr_ifs'].append(1)
    elif is_disc_else(node, source_bytes):
        # if mode == "collect":
        #     node_info['total_ifs'] += 1
        if mode == "modify":
            curr_ifs_clone = mod_ctx['curr_ifs'].copy()
            hash_guard = _extract_disc_hash(node, source_bytes)
            modify_else_statement(
                node,
                source_bytes,
                info,
                modifications,
                counter_list_to_string(curr_ifs_clone[:-1]),
                hash_guard,
            )
    elif is_skip_hash_statement(node):
        if mode == "modify":
            modify_skip_hash_statement(node, source_bytes, info, modifications, counter_list_to_string(mod_ctx['curr_outs']))
    elif is_output_node(node):
        if mode == "collect":
            node_info['total_outs'] += 1
        elif mode == "modify":
            # modify_out_statement(node, source_bytes, info, modifications, counter_list_to_string(mod_ctx['curr_outs']))
            mod_ctx['curr_outs'].append(1)
            # print(mod_ctx['curr_outs'])
    elif is_hash_idx_statement(node):
        if mode == "collect":
            node_info['total_outs'] += 1
        elif mode == "modify":
            modify_hash_idx_statement(node, source_bytes, info, modifications, counter_list_to_string(mod_ctx['curr_outs']))
            mod_ctx['curr_outs'].append(1)
    elif node.type == 'call_expression':
        func_name = next((c for c in node.children if c.type == 'identifier'), None)
        if func_name:
            name = source_bytes[func_name.start_byte:func_name.end_byte].decode('utf8')
            # assert name in info['func_discontinuities'], f"Function {name} not found in discontinuities"
            if name in info['func_discontinuities']:
                disc_func_node = info['func_discontinuities'][name]
                disc_func_total_ifs = disc_func_node['total_ifs']
                disc_func_total_outs = disc_func_node['total_outs']
                print(f"Discontinuous function {name} has {disc_func_total_ifs} ifs and {disc_func_total_outs} outs")
                if disc_func_total_ifs == 0:
                    disc_func_total_ifs = 1
                if disc_func_total_outs == 0:
                    disc_func_total_outs = 1

                if mode == "collect":
                    node_info['total_ifs'] += disc_func_total_ifs
                    node_info['total_outs'] += disc_func_total_outs
                elif mode == "modify":
                    is_start_func = name == 'start'
                    modify_call_expression(node, source_bytes, info, modifications, counter_list_to_string(mod_ctx['curr_ifs']), counter_list_to_string(mod_ctx['curr_outs']), is_start_func)
                    mod_ctx['curr_ifs'].append(disc_func_total_ifs)
                    mod_ctx['curr_outs'].append(disc_func_total_outs)
    
    child_ifs = 0
    child_outs = 0
    for child in node.children:
        ret = process_node_dfs(
            child, 
            source_bytes, 
            info, 
            mode,
            mod_ctx,
            modifications,
            parent_loop=node.id if node.type == 'for_statement' else parent_loop
        )
        if mode == "collect":
            child_info = ret
            if child_info:
                child_ifs += child_info['total_ifs']
                child_outs += child_info['total_outs']
    
    if mode == "collect":
        if is_for_loop(node):
            # Calculate totals including children and loop multipliers
            # (for loops have ifs/ outs only via children)
            node_info['total_ifs'] = child_ifs * node_info['max_iters']
            node_info['total_outs'] = child_outs * node_info['max_iters']
            print(f"For loop {node.id} has {node_info['total_ifs']} ifs and {node_info['total_outs']} outs")
        else:
            node_info['total_ifs'] += child_ifs
            node_info['total_outs'] += child_outs
        info['node_info'][node.id] = node_info
        return node_info
    elif mode == "modify":
        if is_for_loop(node):
            # Now that the loop is complete,
            # first, remove the loop variable multiplier
            # Pop integers and first string multiplier
            while mod_ctx['curr_ifs']:
                if isinstance(mod_ctx['curr_ifs'][-1], str):
                    mod_ctx['curr_ifs'].pop()
                    break
                mod_ctx['curr_ifs'].pop()
            while mod_ctx['curr_outs']:
                if isinstance(mod_ctx['curr_outs'][-1], str):
                    mod_ctx['curr_outs'].pop()
                    break
                mod_ctx['curr_outs'].pop()
            # mod_ctx['curr_ifs'].pop() 
            # mod_ctx['curr_outs'].pop()
            # and then add the total ifs/outs.
            mod_ctx['curr_ifs'].append(info['node_info'][node.id]['total_ifs'])
            mod_ctx['curr_outs'].append(info['node_info'][node.id]['total_outs'])
        return
        # return mod_ctx

def post_process_source(source_code):
    """
    1. Uncomment [Attribute] lines (except [NoDiscontinuity])
    2. Remove [NoDiscontinuity] lines
    3. Pass impl_idx to start function
    """
    lines = source_code.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip [NoDiscontinuity] lines
        if stripped.startswith('//[NoDiscontinuity]'):
            i += 1
            continue
            
        # Uncomment other attribute lines
        elif stripped.startswith('//['):
            processed_lines.append(line.replace('//', '', 1))
            
        # Handle start function call
        elif 'start(x_' in line:
            indent = ' ' * (len(line) - len(line.lstrip()))
            # Modify the start call to use impl_idx directly
            modified_call = line.replace('impl_idx', 'impl_idx')
            processed_lines.append(modified_call)
            
        else:
            processed_lines.append(line)
            
        i += 1
    
    return '\n'.join(processed_lines)

def transform_shader(source_bytes, tree):
    """Transform shader code in a single pass"""
    info = {
        'node_info': {},
        'func_discontinuities': {}
    }
    
    # Find all function definitions and check for cycles
    # The result is topologically sorted
    sorted_functions = find_function_definitions(tree.root_node, source_bytes)

    # Print function continuity information
    print("\nFunction continuity detection:")
    print("-" * 40)
    for func_name, func_node in sorted_functions:
        is_continuous = is_continuous_function(func_node, source_bytes)
        status = "Continuous" if is_continuous else "Discontinuous" 
        print(f"{func_name}: {status}")
    print("-" * 40 + "\n")

    # First pass: Build node info cache
    # Count ifs and outs for discontinuous functions
    # Each function is processed independently, so it is important to do this in topological order respecting dependencies in the DAG
    for func_name, func_node in sorted_functions:
        if not is_continuous_function(func_node, source_bytes):
            node_info = process_node_dfs(func_node, source_bytes, info, mode="collect")
            info['func_discontinuities'][func_name] = {
                'total_ifs': node_info['total_ifs'],
                'total_outs': node_info['total_outs']
            }
    
    # Print if/return count information for discontinuous functions
    print("\nIf/Return count information:")
    print("-" * 40)
    for func_name, stats in info['func_discontinuities'].items():
        print(f"{func_name}:")
        print(f"  Total if statements: {stats['total_ifs']}")
        print(f"  Total outs: {stats['total_outs']}")
    print("-" * 40 + "\n")

    # Second pass: Collect all modifications
    modifications = []
    for func_name, func_node in sorted_functions:
        if not is_continuous_function(func_node, source_bytes):
            mod_ctx = {'curr_ifs':['if_off', 0], 'curr_outs':['out_off', 0]}
            # Don't process the 'run' function
            # if func_name == 'run':
            #     continue
            process_node_dfs(func_node, source_bytes, info, mode="modify", modifications=modifications, mod_ctx_=mod_ctx)
    
    # Add utils.slang import at the beginning of the file
    modifications.append((0, 0, "import utils;\n\n"))

    # Print modifications for each function
    print("\nModifications by function:")
    print("-" * 40)
    
    current_func = None
    for start, end, new_text in sorted(modifications, key=lambda x: x[0]):
        # Find which function this modification belongs to
        for func_name, func_node in sorted_functions:
            if func_node.start_byte <= start <= func_node.end_byte:
                if current_func != func_name:
                    if current_func is not None:
                        print() # Add blank line between functions
                    current_func = func_name
                    print(f"{func_name}:")
                break
        
        # Get the original text
        original = source_bytes[start:end].decode('utf8')
        print(f"  Position {start}-{end}")
        print(f"  Before: {original}")
        print(f"  After:  {new_text}")
    print("-" * 40 + "\n")

    # Apply modifications
    modifications.sort(key=lambda x: x[0], reverse=True)
    result = bytearray(source_bytes)
    for start, end, new_text in modifications:
        result[start:end] = bytes(new_text, 'utf8')
    
    return bytes(result)

# Main code
CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Transform a Slang shader with discontinuity handling.")
    ap.add_argument("src", help="Path to source .slang file")
    ap.add_argument("--dst", help="Destination path (defaults to slang_gen/<name>.slang)", default=None)
    args = ap.parse_args()

    with open(args.src, 'r') as f:
        cpp_source = f.read()

    # Preprocess and parse
    preprocessed_source = preprocess_source(cpp_source) # Comment out Slang.D-specific tags
    source_bytes = bytes(preprocessed_source, "utf8")
    tree = parser.parse(source_bytes)

    # Transform shader in a single pass
    final_source = transform_shader(source_bytes, tree)

    # Post-process and write output
    final_source = post_process_source(final_source.decode('utf8'))
    dst = args.dst or os.path.join("slang_gen", os.path.basename(args.src))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(final_source)


if __name__ == "__main__":
    _main()

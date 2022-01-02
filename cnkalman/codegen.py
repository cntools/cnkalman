import inspect
import sys
import types
from collections.abc import Iterable
from contextlib import redirect_stdout
from typing import NamedTuple
import math

import symengine as sp
from symengine import Pow, cse, Mul
from sympy import evaluate

import sympy

def isinstance_namedtuple(obj) -> bool:
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )

def flatten_args(bla, prefix=''):
    output = []
    if hasattr(bla, "__iter__"):
        for i, item in enumerate(bla):
            output += flatten_args(item, prefix=prefix+'['+str(i)+']') if hasattr(item, "__iter__") or isinstance_namedtuple(item) else [(prefix+"["+str(i)+"]", item)]
    elif hasattr(bla, "__dict__"):
        for k,vall in bla.__dict__.items():
            if hasattr(vall, "__iter__") or isinstance_namedtuple(vall):
                for ik, iv in flatten_args(vall, prefix=prefix+'.'+k):
                    output.append((ik, iv))
            else:
                output.append((prefix + "." + k, vall))

    return output


def make_sympy(expressions):
    flatten = []

    if type(expressions) == sp.MutableDenseMatrix:
        return expressions

    if hasattr(expressions, "atoms"):
        return [expressions]

    if isinstance(expressions, list):
        return sp.MutableDenseMatrix(expressions)

    if not hasattr(expressions, "_sympy_"):
        for col in expressions:
            if hasattr(col, '_sympy_'):
                flatten.append(col)
            else:
                for cell in col:
                    flatten.append(cell)
    else:
        flatten.append(expressions)
    return flatten

def expand_pow(x):
    pass

def number(x):
    if x.is_Number:
        return float(x)
    return None

def clean_parens(txt):
    if txt[0] == '(' and txt[-1] == ')':
        cnt = 0
        for c in txt[1:-1]:
            if c == '(':
                cnt += 1
            if c == ')':
                cnt -= 1
            if cnt < 0:
                return txt
        return clean_parens(txt[1:-1])
    return txt

def ccode_wrapper(item, depth = 0):
    #return sp.ccode(item)

    if item.is_Atom:
        if item == True:
            return "true"
        if item == False:
            return "false"
        return sp.ccode(item)

    newargs = list(map(lambda x: ccode_wrapper(x, depth+1), item.args))

    infixes = {
        Mul: '*',
        sp.Add: '+',
        sp.GreaterThan: '>=',
        sp.StrictGreaterThan: '>',
        sp.LessThan: '<=',
        sp.StrictLessThan: '<'
    }

    if item.__class__ == Pow:
        # Basically it's always faster to never call pow
        if item.args[1].is_Number and abs(item.args[1]) < 20:
            invert = item.args[1] < 0
            num, den = abs(item.args[1]).get_num_den()

            if den == 1 or den == 2:
                mul_cnt = num if den == 1 else (num - 1) / 2
                muls = [newargs[0]] * int(mul_cnt)
                if den == 2:
                    muls.append("sqrt(" + clean_parens(newargs[0]) + ")")
                v = " * ".join(muls)
                if len(muls) > 1:
                    v = "(" + v + ")"
                if invert:
                    v = "(1. / " + v + ")"
                return v
        return "pow(%s, %s)" % tuple(newargs)
    elif item.__class__ in infixes:
        return "(" + (" " + infixes[item.__class__] + " ").join(newargs) + ")"
    elif item.__class__ == sp.Piecewise:
        if item.args[1] == True:
            return newargs[0]
        if item.args[1] == False:
            return newargs[2]
        return "(%s ? %s : %s)" % (newargs[1], newargs[0], newargs[2])

    return item.__class__.__name__ + "(" + ", ".join(map(clean_parens, newargs)) + ")"
    #raise Exception("Unhandled type " + item.__class__.__name__)

def ccode(item):
    return clean_parens(ccode_wrapper(item))

class WrapTuple:
    def __init__(self, n, t):
        self.n = n
        self.t = t

    def __getitem__(self, item): return self.t[item]
    def __iter__(self): yield from self.t
    def __str__(self): return self.n
    def __repr__(self): return self.n

def get_argument(n, argument_specs):
    if n in argument_specs:
        a = argument_specs[n]
        if isinstance(a, tuple):
            return WrapTuple(n, a)
        if isinstance(a, int):
            return WrapTuple(n, [ sympy.symbols(f"{n}{i}") for i in range(a)])
        return a
    if n in globals():
        return globals()[n]
    return sympy.symbols(n)

def get_name(a):
    if type(a) == list:
        return "_".join(map(get_name, a))
    if hasattr(a, '__name__'):
        return a.__name__
    return str(a)

import inspect, ast


def flatten_func(func, name=None, args=None, suffix = None, argument_specs ={}):
    if callable(func):
        name = func.__name__
        args = [get_argument(n, argument_specs) for n in inspect.getfullargspec(func).args]

    if suffix is not None:
        name = name + "_" + suffix

    if isinstance(func, types.FunctionType):
        try:
            func = func(*map_arg(args))
        except Exception as e:
            sys.stderr.write(f"Error evaluating {name}. Likely a variable needs a length annotation: {e}\n")
            return None, None

    return make_sympy(func), args

def get_type(a):
    if callable(a):
        return get_type(a())
    if hasattr(a, "__iter__"):
        ty = get_type(a[0])
        if ty[-1] != "*":
            ty += "*"
        return ty
    if isinstance_namedtuple(a):
        return a.__class__.__name__ + "*"
    return "FLT"

def arg_str(arg):
    a = arg[1]
    return "const %s %s" % (get_type(a), get_name(a))

def generate_args_string(args, as_call = False):
    return ", ".join(map(lambda x: get_name(x[1]) if as_call else arg_str, enumerate(args)))

def generate_ccode(func, name=None, args=None, suffix = None, argument_specs ={}, outputs = [('out', -1)], preamble = ""):
    flatten, args = flatten_func(func, name, args, suffix, argument_specs)
    if flatten is None:
        return None

    if callable(func):
        name = func.__name__
        args = [get_argument(n, argument_specs) for n in inspect.getfullargspec(func).args]

    if suffix is not None:
        name = name + "_" + suffix

    singular_return = len(flatten) == 1

    cse_output = cse(sp.Matrix(flatten))

    if singular_return:
        print("static inline FLT gen_%s(%s) {" % (name, ", ".join(map(arg_str, enumerate(args)))))
    else:
        print("static inline void gen_%s(%s, %s) {" % (name, ", ".join(["CnMat* " + s[0] for s in outputs]), ", ".join(map(arg_str, enumerate(args)))))

    if preamble:
        print(preamble.strip("\r\n"))

    free_symbols = {k.__str__() for k in flatten[0].free_symbols} if isinstance(flatten, list) else {k.__str__() for k in flatten.free_symbols}
    # Unroll struct types
    for idx, a in enumerate(args):
        if callable(a):
            name = get_name(a)
            for k, v in flatten_args(a()):
                if f"{name}{k.strip('[]')}" in free_symbols:
                    print("\tconst FLT %s = %s%s;" % (str(v), "(*"+name+")" if isinstance_namedtuple(a()) else name, k))
        elif isinstance(a, WrapTuple):
            name = get_name(a)
            for k, v in flatten_args(a.t):
                if f"{name}{k.strip('[]')}" in free_symbols:
                    print("\tconst FLT %s = %s%s;" % (str(v), name, k))

    print("\n".join(
        map(lambda item: "\tconst FLT %s = %s;" % (
            sp.ccode(item[0]), ccode(item[1]).replace("\n", " ").replace("\t", " ")), cse_output[0])))

    output_idx = 0
    outputs_idx = 0
    needs_guard = len(outputs) > 1
    tabs = "\t"
    for item in cse_output[1]:
        current_shape = outputs[outputs_idx][1] if isinstance(outputs[outputs_idx][1], tuple) else [outputs[outputs_idx][1], 1]
        current_row = output_idx // current_shape[1]
        current_col = output_idx % current_shape[1]
        if hasattr(item, "tolist"):
            for item1 in sum(item.tolist(), []):
                print("\tcnMatrixSet(%s, %d, %d, %s);" % (outputs[outputs_idx][0], current_row, current_col, output_idx, ccode(item1).replace("\n", " ").replace("\t", " ")))
                output_idx += 1
                current_row = output_idx / current_shape[1]
                current_col = output_idx % current_shape[1]
        else:
            if singular_return:
                print("\treturn %s;" % (ccode(item).replace("\n", " ").replace("\t", " ")))
            else:
                print("\tcnMatrixSet(%s, %d, %d, %s);" % (outputs[outputs_idx][0], current_row, current_col, ccode(item).replace("\n", " ").replace("\t", " ")))
            output_idx += 1
        if output_idx >= math.prod(current_shape) > 0:
            outputs_idx += 1
            output_idx = 0

    print("}")
    print("")
    return flatten


def jacobian(v, of):
    if hasattr(v, 'jacobian'):
        return v.jacobian(sp.Matrix(of))
    return sp.Matrix([v]).jacobian(sp.Matrix(of))

def map_arg(arg):
    if callable(arg):
        return map_arg(arg())
    elif isinstance(arg, list):
        return list(map(map_arg, arg))
    elif isinstance(arg, tuple):
        return tuple(map(map_arg, arg))
    return arg

def flat_values(a):
    if isinstance(a, Iterable):
        return sum([flat_values(it) for it in a], [])
    if hasattr(a, '__dict__'):
        return flat_values(a.__dict__.values())
    return [a]


def generate_jacobians(func, suffix=None,transpose=False,jac_all=False, jac_over=None, argument_specs={}):
    rtn = {}

    fx, _= flatten_func(func, argument_specs=argument_specs)

    func_args = [get_argument(n, argument_specs) for n in inspect.getfullargspec(func).args]
    jac_of = {}
    if jac_over is not None:
        jac_of[get_name(jac_over)] = flat_values(map_arg(jac_over))
    else:
        jac_of.update({get_name(arg): flat_values(map_arg(arg)) for arg in func_args})

    if jac_all:
        jac_of['all'] = sum(list(jac_of.values()), [])

    feval = (func(*map_arg(func_args)))

    for name, jac_value in jac_of.items():
        fname = func.__name__  + '_jac_' + name
        if type(feval) == list or type(feval) == tuple:
            feval = sp.MutableDenseMatrix(feval)
        this_jac = jacobian(feval, jac_value)
        if transpose:
            this_jac = this_jac.transpose()

        jac_shape = this_jac.shape
        jac_size = this_jac.shape[0] * this_jac.shape[1]

        print("// Jacobian of", func.__name__, "wrt", jac_value)
        generate_ccode(this_jac, fname, func_args, suffix=suffix, outputs=[('Hx', jac_shape)])

        fxm = sp.MutableDenseMatrix(fx)
        fx_size = fxm.shape[0] * fxm.shape[1]
        jac_with_hx = this_jac.reshape(jac_size, 1).col_join(fxm.reshape(fx_size, 1))

        print("// Full version Jacobian of", func.__name__, "wrt", jac_value)

        fn_suffix = ""
        if suffix is not None:
            fn_suffix = "_" + suffix

        gen_call = f"gen_{func.__name__}{fn_suffix}(hx, {generate_args_string(func_args, True)});"
        if fx_size == 1:
            gen_call = f"hx->data[0] = gen_{func.__name__}{fn_suffix}({generate_args_string(func_args, True)});"

        preamble = f"""
    if(Hx == 0) {{ 
        {gen_call}
        return;
    }}
    if(hx == 0) {{ 
        gen_{fname}{fn_suffix}(Hx, {generate_args_string(func_args, True)});
        return;
    }}"""
        generate_ccode(jac_with_hx, fname + "_with_hx", func_args, suffix=suffix, outputs=[('Hx', jac_shape), ('hx', this_jac.shape[0] - jac_size)], preamble=preamble)

        rtn['jacobian_of_' + name] = this_jac.reshape(*jac_shape)
    return rtn, func_args

def generate_code_and_jacobians(f,transpose=False, jac_over=None, argument_specs = {}):
    generate_ccode(f, argument_specs = argument_specs)
    return generate_jacobians(f, argument_specs = argument_specs, transpose=transpose, jac_over=jac_over)

from pathlib import Path

generate_code_files = {}
def get_file(fn):
    if not '--cnkalman-generate-source' in sys.argv:
        return None
    if fn in generate_code_files:
        return generate_code_files[fn]
    path = Path(fn)
    generate_code_files[fn] = open(f"{path.parent.as_posix()}/{path.stem}.gen.h", 'w')
    return generate_code_files[fn]

import numpy as np
def functionify(args_info, jac):
    def f(*args):
        subset = {}
        for i, info in enumerate(args_info):
            if isinstance(info, WrapTuple):
                for j, s in enumerate(info.t):
                    v = args[i][j]
                    subset[s.__str__()] = v
            else:
                subset[info.__str__()] = args[i]

        return np.array(jac.subs(subset)).astype(np.float64)
    return f

def generate_code(**kwargs):

    def f(func):
        f = get_file(inspect.getfile(func))
        g = lambda *args: np.array(func(*args), dtype=np.float64)
        with redirect_stdout(f):
            jacs, args = generate_code_and_jacobians(func, argument_specs=kwargs)
            for k, v in jacs.items():
                setattr(g, k, functionify(args, v))
        return g
    return f

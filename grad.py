import symengine as se
from symengine import sympify, cse, ccode, sympify, Basic

from jinja2 import Template


class CCodePrinter:

    def doprint(self, expr, assign_to=None):
        if not isinstance(assign_to, (Basic, type(None), str)):
            raise TypeError("{} cannot assign to object of type {}".format(
                    type(self).__name__, type(assign_to)))

        expr = sympify(expr)
        if not assign_to:
            if expr.is_Matrix:
                raise RuntimeError("Matrices need a assign_to parameter")
            return ccode(expr)

        assign_to = str(assign_to)
        if not expr.is_Matrix:
            return f"{assign_to} = {ccode(expr)};"

        code_lines = []
        for i, element in enumerate(expr):
            code_line = f'{assign_to}[{i}] = {element};'
            code_lines.append(code_line)
        return '\n'.join(code_lines)

class CUDAPrinter:

	def __init__(self):
		self.cprinter = CCodePrinter()

	def tcs_f(self, sym_expr):
		sstr = self.cprinter.doprint(sym_expr)
		sstr.replace('exp(', 'expf(')
		sstr.replace('cos(', 'cosf(')
		sstr.replace('sin(', 'sinf(')
		sstr.replace('tan(', 'tanf(')
		sstr.replace('acos(', 'acosf(')
		sstr.replace('asin(', 'asinf(')
		sstr.replace('atan(', 'atanf(')
		sstr.replace('cosh(', 'coshf(')
		sstr.replace('sinh(', 'sinhf(')
		sstr.replace('tanh(', 'tanhf(')
		sstr.replace('acosh(', 'acoshf(')
		sstr.replace('asinh(', 'asinhf(')
		sstr.replace('atanh(', 'atanhf(')
		sstr.replace('sqrt(', 'sqrtf(')
		sstr.replace('rsqrt', 'rsqrtf')
		return sstr

	def tcs_d(self, sym_expr):
		sstr = self.cprinter.doprint(sym_expr)
		return sstr


#expr = sympify('S0*(f*exp(-b*D_1) + (1-f)*exp(-b*D_2))')

#def walk_expr(expr):
#	print(expr.args)
#	if len(expr.args) > 0:
#		walk_expr(expr.args[0])
#	if len(expr.args) > 1:
#		walk_expr(expr.args[1])


#walk_expr(expr)

def res(expr, pars, consts):
	resexpr = sympify(expr)
	substs, reduced = cse([resexpr])
	return (substs, reduced)

def jac(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = []
	for e in pars:
		exprs.append(resexpr.diff(e))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def hes(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = []
	for i in range(0,len(pars)):
		for j in range(0,i+1):
			exprs.append(resexpr.diff(pars[i]).diff(pars[j]))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def res_jac(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = [resexpr]
	for e in pars:
		exprs.append(resexpr.diff(e))

	substs, reduced = cse(exprs)
	return (substs, reduced)

def res_jac_hes(expr, pars, consts):
	resexpr = sympify(expr)

	exprs = [resexpr]
	for e in pars:
		exprs.append(resexpr.diff(e))

	for i in range(0,len(pars)):
		for j in range(0,i+1):
			exprs.append(resexpr.diff(pars[i]).diff(pars[j]))

	substs, reduced = cse(exprs)
	return (substs,  reduced)

expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))+D_1**2'
expr_sym = sympify(expr)

#print(CCodePrinter().doprint(expr_sym))

res_jac_hes_temp = Template(
"""
__device__
void {{funcid}}(const {{fp_type}}* params, const {{fp_type}}* consts, 
	const {{fp_type}}* data, const {{fp_type}}* weights, float lambda, {{fp_type}}* res, 
	{{fp_type}}* jac, {{fp_type}}* hes, {{fp_type}}* lam_hes) 
{
	{{zero_mat_funcid}}(hes);

	for (int i = 0; i < {{ndata}}; ++i) {
{{sub_expr}}

{{res_expr}}

{{jac_expr}}

{{hes_expr}}
	}

	for (int i = 1; i < {{npar}}; ++i) {
		for (int j = 0; j < i; ++j) {
			hes[j*{{npar}} + i] = hes[i*{{npar}} + j];
		}
	}

	{{mul_transpose_diag_mat_funcid}}(jac, weights, lam_hes);
	{{add_mat_mat_ldiag_funcid}}(hes, lamda, lam_hes);
}
""")

pars = ['S0', 'f', 'D_1', 'D_2']
consts = ['b']

substs, reduced = res_jac_hes(expr, pars, consts)

cuprint = CUDAPrinter()
fp_type_str = 'float'



sub_str = ""
print('substr')
for s in substs:
	sub_str += '\t\t'+fp_type_str+' '+cuprint.tcs_f(s[0])+' = '+cuprint.tcs_f(s[1])+';\n'
print(sub_str)

res_str = ""
print('res')
for s in reduced[:1]:
	res_str += '\t\tres[i] = '+cuprint.tcs_f(s)+';'
print(res_str)

jac_str = ""
print('jac')
i = 0
for s in reduced[1:(len(pars) + 1)]:
	jac_str += '\t\tjac[i*'+str(len(pars))+'+'+str(i)+'] = '+cuprint.tcs_f(s)+';\n'
	i += 1
print(jac_str)

hes_str = ""
print('hes')
k = 0
for i in range(0,len(pars)):
	for j in range(0,i+1):
		s = reduced[len(pars) + k + 1]
		hes_str += '\t\thes['+str(i)+'*'+str(len(pars))+'+'+str(j)+'] += '+cuprint.tcs_f(s)+';\n'
		k += 1
print(hes_str)
	

rjh_kernel = res_jac_hes_temp.render(sub_expr=sub_str, res_expr=res_str, 
	jac_expr=jac_str, hes_expr=hes_str)

print(rjh_kernel)


#print('res')
#res(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])

#print('jac')
#jac(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])

#print('res_jac_hes')
#hes(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])



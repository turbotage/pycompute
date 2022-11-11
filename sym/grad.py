import symengine as se
from symengine import sympify, cse, ccode, sympify, Basic



#expr = sympify('S0*(f*exp(-b*D_1) + (1-f)*exp(-b*D_2))')

#def walk_expr(expr):
#	print(expr.args)
#	if len(expr.args) > 0:
#		walk_expr(expr.args[0])
#	if len(expr.args) > 1:
#		walk_expr(expr.args[1])


#walk_expr(expr)


expr = 'S0*(f*exp(-b*D_1)+(1-f)*exp(-b*D_2))+D_1**2'
expr_sym = sympify(expr)

#print(CCodePrinter().doprint(expr_sym))



pars = ['S0', 'f', 'D_1', 'D_2']
consts = ['b']

substs, reduced = res_jac_hes(expr, pars, consts)

cuprint = CUDAPrinter()
fp_type_str = 'float'



rjh_kernel = res_jac_hes_temp.render(sub_expr=sub_str, res_expr=res_str, 
	jac_expr=jac_str, hes_expr=hes_str)

print(rjh_kernel)


#print('res')
#res(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])

#print('jac')
#jac(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])

#print('res_jac_hes')
#hes(expr, ['S0', 'f', 'D_1', 'D_2'], ['b'])



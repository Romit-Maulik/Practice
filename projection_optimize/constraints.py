'''
Define constraints (depends on your problem)
https://stackoverflow.com/questions/42303470/scipy-optimize-inequality-constraint-which-side-of-the-inequality-is-considered
[0.1268 0.467 0.5834 0.2103 -0.1268 -0.5425 -0.5096 0.0581] . The bounds are +/-30% of this.
''' 
t_base = [0.1268, 0.467, 0.5834, 0.2103, -0.1268, -0.5425, -0.5096, 0.0581]
t_lower = [0.08876, 0.3269, 0.40838, 0.14721, -0.1648, -0.70525, -0.66248, 0.04067]
t_upper = [0.1648, 0.6071, 0.75842, 0.27339, -0.08876, -0.37975, -0.35672, 0.07553]

def f_factory(i):
    def f_lower(t):
        return t[i] - t_lower[i]

    def f_upper(t):
        return -t[i] + t_upper[i]
    return f_lower, f_upper

functions = []
for i in range(len(t_base)):
    f_lower, f_upper = f_factory(i)
    functions.append(f_lower)
    functions.append(f_upper)

cons=[]
for ii in range(len(functions)):
    # the value of ii is set in each loop
    cons.append({'type': 'ineq', 'fun': functions[ii]})


if __name__ == '__main__':
    print('Constraints file')
'''
Define constraints (depends on your problem)
https://stackoverflow.com/questions/42303470/scipy-optimize-inequality-constraint-which-side-of-the-inequality-is-considered
[0.1268 0.467 0.5834 0.2103 -0.1268 -0.5425 -0.5096 0.0581] . The bounds are +/-30% of this.
''' 
t_base = [0.1268, 0.467, 0.5834, 0.2103, -0.1268, -0.5425, -0.5096, 0.0581]
t_lower = [-0.08876, -0.3269, -0.40838, -0.14721, 0.1648, 0.70525, 0.66248, -0.04067]
t_upper = [0.1648, 0.6071, 0.75842, 0.27339, -0.08876, -0.37975, -0.35672, 0.07553]

def con_0_lower(t):
    return t[0]+t_lower[0]
def con_0_upper(t):
    return -t[0]+t_upper[0]

def con_1_lower(t):
    return t[1]+t_lower[1]
def con_1_upper(t):
    return -t[1]+t_upper[1]

def con_2_lower(t):
    return t[2]+t_lower[2]
def con_2_upper(t):
    return -t[2]+t_upper[2]

def con_3_lower(t):
    return t[3]+t_lower[3]
def con_3_upper(t):
    return -t[3]+t_upper[3]

def con_4_lower(t):
    return t[4]+t_lower[4]
def con_4_upper(t):
    return -t[4]+t_upper[4]

def con_5_lower(t):
    return t[5]+t_lower[5]
def con_5_upper(t):
    return -t[5]+t_upper[5]

def con_6_lower(t):
    return t[6]+t_lower[6]
def con_6_upper(t):
    return -t[6]+t_upper[6]

def con_7_lower(t):
    return t[7]+t_lower[7]
def con_7_upper(t):
    return -t[7]+t_upper[7]

cons = ({'type':'ineq', 'fun': con_0_lower},\
        {'type':'ineq', 'fun': con_0_upper},\
        {'type':'ineq', 'fun': con_1_lower},\
        {'type':'ineq', 'fun': con_1_upper},\
        {'type':'ineq', 'fun': con_2_lower},\
        {'type':'ineq', 'fun': con_2_upper},\
        {'type':'ineq', 'fun': con_3_lower},\
        {'type':'ineq', 'fun': con_3_upper},\
        {'type':'ineq', 'fun': con_4_lower},\
        {'type':'ineq', 'fun': con_4_upper},\
        {'type':'ineq', 'fun': con_5_lower},\
        {'type':'ineq', 'fun': con_5_upper},\
        {'type':'ineq', 'fun': con_6_lower},\
        {'type':'ineq', 'fun': con_6_upper},\
        {'type':'ineq', 'fun': con_7_lower},\
        {'type':'ineq', 'fun': con_7_upper},)

if __name__ == '__main__':
    print('Constraints file')
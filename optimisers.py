import numpy as np

A=np.diag([1,10,100]) # test A matrix for Hessian of quadratic
center = np.array([2.,4.,-8.]) # solution of least squares problem

def get_value(x):
    # x is numpy array with shape (d,). In this example d=3.
    global A
    global center
    return 0.5*np.dot(x - center, np.dot(A, x -  center)) 
    # this works because python automatically interprets a d-dimensional vector as either a column vector or row
    # vector appropriately. If you want you can be more explicit and give the following instead:
    # return 0.5*np.dot(x[None,:], np.dot(A,x[:,None])) 
    
def get_gradient(x):
    # x is numpy array with shape (d,). In this example d=3.    
    global A
    global center    
    return np.dot(A,x - center)
    # this works because python automatically interprets a d-dimensional vector as either a column vector or row
    # vector appropriately. If you want you can be more explicit and give the following instead:
    # return np.dot(A,x[:,None])[:,0]
    
def descent_update(x, desc_dir, eta):
    """ 
    x: is a d-dimensional vector giving the current iterate.
       It is represented as numpy array with shape (d,)
    desc_dir: is a d-dimensional vector giving the descent direction, the next iterate
             is along this direction.
             It is represented as numpy array with shape (d,)
    eta: is a float value indicating the step size
    Returns
    
    Next iterate with the given step size.
    Should be represented by numpy array with shape (d,)
    """
    return x + eta*desc_dir

def descent_update_AG(x, desc_dir, alpha = 0.25, beta = 0.5):
    """ 
    x: is a d-dimensional vector giving the current iterate.
       It is represented as numpy array with shape (d,)
    desc_dir: is a d-dimensional vector giving the descent direction, the next iterate
             is along this direction.
             It is represented as numpy array with shape (d,)
    Returns
    
    Next iterate where step size is chosen to satisfy Armijo-Goldstein conditions.
    Should be represented by numpy array with shape (d,)
    """
    min_eta = 0.0
    max_eta = np.inf
    max_iter = 1e2
    eta = 1
    f = 0
    i = 0
    grad = get_gradient(x)
    val = get_value(x) 
    grad_comp = (grad@desc_dir)

    while True:
        i = i+1
        f = get_value(x + eta*desc_dir)
        f1 = val + beta*eta*grad_comp
        f2 = val + alpha*eta*grad_comp
       
        if f<f1:
            min_eta = eta
            eta = np.min([2*eta, (min_eta+max_eta)/2])
        elif f>f2:
            max_eta = eta
            eta = (min_eta + max_eta)/2
        elif (f1<f and f<f2):
            break   
        
        if(i>max_iter):
            break
    return descent_update(x, desc_dir, eta)

def descent_update_FR(x, desc_dir):
    """ 
    x: is a d-dimensional vector giving the current iterate.
       It is represented as numpy array with shape (d,)
    desc_dir: is a d-dimensional vector giving the descent direction, the next iterate
             is along this direction.
             It is represented as numpy array with shape (d,)

    Returns
    Next iterate where step size is chosen to satisfy full relaxation conditions (approximately)
    Should be represented by numpy array with shape (d,)
    """
    g_dash = lambda eta: get_gradient(x+eta*desc_dir)@desc_dir
    min_eta = 0
    max_eta = np.inf
    thresh = 1e-15
    eta = 1
    i = 1
    max_iter = 1e2
    while(True):
        i = i+1

        if i>max_iter:
            break   

        if g_dash(eta) > thresh:
            max_eta = eta
            eta = (max_eta+min_eta)/2
        elif g_dash(eta)< - thresh:
            min_eta = eta
            eta = np.min([2*eta, (min_eta+max_eta)/2])

        elif np.abs(g_dash(eta))<thresh:
            break
    return descent_update(x, desc_dir, eta)

def BFGS_update(H, s, y):
    """ Returns H_{k+1} given H_k and s and y based on BFGS update.
    H: numpy array with shape (d,d)
    s: numpy array with shape (d,)
    y: numpy array with shape (d,)
    """
    s = np.reshape(s, (-1,1))
    y = np.reshape(y, (-1,1))
    eps = 1e-15 # epsilon to prevent nan vals for rho
    rho = float(1/(s.T@y+eps))
    I = np.eye(s.shape[0])
    return (I - rho*(s@y.T))@H@(I - rho*y@s.T) + rho*s@s.T

def gradient_descent(x0, num_iter=100, eta='AG'):
    """Runs gradient descent till convergence or till number of iterations
    
    x0: Initial point , represented by numpy array with shape (d,)
    num_iter: number of iterations to run
    eta: The rule by which step size is set. It can take the string values of "AG" or "FR" 
         corresponding to Armijo-Goldstein and Full relaxation criteria. It can also take a
         positive real value, in which case the step size is set to that constant.
    
    Returns:
    Final iterate which is a d-dimensional vector represented by numpy array with shape (d,). 
    The algorithm can be stopped if either  the number of iterations is reached
    or if the change in iterates is less than tol. i.e. ||x_{t+1} -x_{t}||<=tol.
    
    """
    x = x0
    for i in range(num_iter):
        if eta=='AG':
            x = descent_update_AG(x, -get_gradient(x))
        elif eta=='FR':
            x = descent_update_FR(x, -get_gradient(x))
        else:
            x = descent_update(x, -get_gradient(x), eta)
    return x

def quasi_Newton(x0, H0, num_iter=100, eta='AG'):
    """Runs Quasi Newton with BFGS till convergence or till number of iterations.
    
    x0: Initial point , represented by numpy array with shape (d,)
    H0: Initial inverse Hessian estimate. Represented by numpy array with shape (d,d)
    num_iter: number of iterations to run
    eta: The rule by which step size is set. It can take the string values of "AG" or "FR" 
         corresponding to Armijo-Goldstein and Full relaxation criteria. It can also take a
         positive real value, in which case the step size is set to that constant.
    
    Returns:
    Final iterate which is a d-dimensional vector represented by numpy array with shape (d,). 
    The algorithm can be stopped if either  the number of iterations is reached
    or if the change in iterates is less than tol. i.e. ||x_{t+1} -x_{t}||<=tol.
    
    """
    x = x0
    x_previous = x0
    H = H0
    for i in range(num_iter):
        if eta == 'AG':
            x = descent_update_AG(x, -1*H@get_gradient(x))
        elif eta == 'FR':
            x = descent_update_FR(x, -1*H@get_gradient(x))      
        else:
            x = descent_update(x, -1*H@get_gradient(x), eta)
        
        if(i<num_iter):
            s = x - x_previous
            y = get_gradient(x) - get_gradient(x_previous)
            H = BFGS_update(H, s, y)
            x_previous = x
    return x
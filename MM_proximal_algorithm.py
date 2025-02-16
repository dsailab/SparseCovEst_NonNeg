import numpy  as np
class PDTE_NN_FC(object):
    def __init__(self,x,s,t,fhi,gamma,lambda_val,a_val,dimension):
        self.x=x
        self.s=s #sampe covariance
        self.t=t #tau
        self.fhi=fhi #step size 
        self.gamma=gamma
        self.lambda_val=lambda_val
        self.a_val=a_val
        self.dimension=dimension
    def f_function(self,x,w):
        return 0.5*(np.linalg.norm(x-self.s+w)**2)-self.t*np.log(np.linalg.det(x))
    def gradient(self,x,w):
        inv=np.linalg.inv(x)
        inv_x=0.5*(inv+inv.T)
        return x-self.s+w-self.t*inv_x
    def mcp_derivative(self,x):
        return (self.lambda_val-abs(x)/self.a_val)*(abs(x)<=self.lambda_val*self.a_val)+0*(abs(x)>self.lambda_val*self.a_val)    
    def mcp_penalty(self, x):
        is_linear = (np.abs(x) <= self.lambda_val*self.a_val)
        is_constant = (self.a_val * self.lambda_val) < np.abs(x)
        linear_part = (self.lambda_val*abs(x)-(x**2)/(2*self.a_val) )* is_linear
        constant_part =  (0.5*(self.lambda_val**2)*self.a_val)* is_constant
        return linear_part + constant_part
    def update_w(self,x):
        scad_matrix=self.mcp_derivative(x)
        return scad_matrix-np.diag(scad_matrix.diagonal())
    
    def function_value(self,x):
        return (1/2)*np.linalg.norm(x-self.s,'fro')**2-self.t*np.log(np.linalg.det(x))-np.trace(self.mcp_penalty(x))+np.sum(self.mcp_penalty(x))
    def MM_process(self,x):
        value_list=[]
        x=self.x
        for k in range(0,100):
            iteration=0
            flag1=True
            x_outer_old=x
            w=self.update_w(x_outer_old)
            while flag1:
                flag2=True
                iteration=iteration+1
                x_old=x
                if iteration==1:
                    fhi_t=self.fhi
                else:
                    fhi_t=max(self.fhi, (1/self.gamma)*fhi_t)
                while flag2:
                    x=x_old-(1/fhi_t)*self.gradient(x_old,w)
                    x[x<0]=0
                    a1=self.f_function(x_old,w)
                    a2=np.sum(self.gradient(x_old,w)*(x-x_old))
                    a3=np.linalg.norm(x-x_old)**2
                    g_value=a1+a2+0.5*fhi_t*a3
                    f_value=self.f_function(x,w)
                    if g_value>=f_value and min(np.linalg.eigh(x)[0])>0 :
                        x_new=x
                        break
                    else:
                        fhi_t=self.gamma*fhi_t
                value_list.append(self.function_value(x_new))
                if np.linalg.norm(x_new-x_old)/np.linalg.norm(x_old)<1e-6:
                    print('inner layer convergence')
                    break
            if np.linalg.norm(x_new-x_outer_old)/np.linalg.norm(x_outer_old)<1e-6:
                print('out layer convergence')
                return (x_new, value_list)
        return (x_new,value_list)
                    
                    
                
            
    
    
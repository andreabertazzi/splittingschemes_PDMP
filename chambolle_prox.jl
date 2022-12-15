"""
f = chambolle_prox_TV(g, lambda, maxiter)
Proximal  point operator for the TV regularizer
Uses the Chambolle's projection  algorithm:
A. chambolle_TV, "An Algorithm for Total Variation Minimization and
Applications", J. Math. Imaging Vis., vol. 20, pp. 89-97, 2004.
Optimization problem:
    arg min = (1/2) || y - x ||_2^2 + lambda TV(x)
        x
=========== Required inputs ====================
'g'       : noisy image (size X: ny * nx)
'lambda'  : regularization  parameter according
'maxiter' :maximum number of iterations
"""

# The divergence function that the prox_TV function needs
function DivergenceIm(p1,p2)
    z = p2[:,2:end-1] - p2[:,1:end-2]
    v = [p2[:,1] z -p2[:,end]]
    z = p1[2:end-1, :] - p1[1:end-2,:]
    u = [p1[1,:] z' -p1[end,:]]
    u = u'
    return v + u
end
# The Gradient of the image that prox_TV function needs
function GradientIm(u)
    z = u[2:end, :] - u[1:end-1,:]
    dux = [z' zeros(size(z,2))]
    dux = dux'
    z = u[:,2:end] - u[:,1:end-1]
    duy = [z zeros((size(z,1),1))]
    return  dux,duy
end
# total variation proximal operator
# inputs:
# g: image
# apprParam: the approximation parameter of the proximal algorithm (\lambda)
# MaxIter: number of iterations, normally if you set between 20 or 25 iterations
# 			you will have a very good approximation.
# output:
# the total-variation Prox operator of the image 'g'
function chambolle_prox_TV(g,apprParam,MaxIter)
    # initialize
    px = zeros((size(g,1),size(g,2)))
    py = zeros((size(g,1),size(g,2)))
    cont = true
    k    = 0
    tau = 0.249

    while cont
        k = k+1
        # compute Divergence of (px, py)
        divp = DivergenceIm(px,py)
        u = divp - g/apprParam
        # compute gradient
        upx,upy = GradientIm(u)
        tmp = sqrt.(upx.*upx .+ upy.*upy)
        #print(typeof(px),typeof(upx))
        px = (px .+ tau * upx)./(1 .+ tau * tmp)
        py = (py .+ tau * upy)./(1 .+ tau * tmp)
        cont = (k<MaxIter)
    end

    return g - apprParam * DivergenceIm(px,py)
end

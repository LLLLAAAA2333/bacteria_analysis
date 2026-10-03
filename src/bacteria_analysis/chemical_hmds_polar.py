"""Stable radial/angular HMDS coordinates for concentrated chemical distances."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import null_space


def polar_start(euclidean, scale):
    points = euclidean - euclidean.mean(axis=0)
    radius = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    columns = [np.log(np.maximum(radius, 1e-10)), azimuth]
    if points.shape[1] == 3:
        columns.append(np.arccos(np.clip(points[:, 2]/radius, -1, 1)))
    return np.r_[np.column_stack(columns).ravel(), np.log(scale)]


def polar_distances(x, n, dimension):
    """Distances in input-scaled units plus their exact coordinate Jacobian.

    Hyperbolic radius = lambda * exp(log input-unit radius).  The pairwise
    sinh expression avoids subtracting nearly equal large cosh products.
    """
    # The origin is a virtual point, not an observed bacterium. Its pairs are excluded.
    n = n+1
    parameters = x[:-1].reshape(n-1, dimension)
    scale = np.exp(x[-1])
    radius = scale * np.exp(parameters[:, 0])
    theta = parameters[:, 1]
    if dimension == 2:
        direction = np.column_stack([np.cos(theta), np.sin(theta)])
        derivatives = np.stack([np.column_stack([-np.sin(theta), np.cos(theta)])], axis=1)
    else:
        phi = parameters[:, 2]
        direction = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
        dt = np.column_stack([-np.sin(phi)*np.sin(theta), np.sin(phi)*np.cos(theta), np.zeros(n-1)])
        dp = np.column_stack([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), -np.sin(phi)])
        derivatives = np.stack([dt, dp], axis=1)
    pi, pj = np.triu_indices(n, 1)
    predicted = np.zeros(len(pi))
    jacobian = np.zeros((len(pi), len(x)))
    anchored = pi == 0
    predicted[anchored] = radius[pj[anchored]-1]/scale
    jacobian[np.flatnonzero(anchored), (pj[anchored]-1)*dimension] = predicted[anchored]
    rows = np.flatnonzero(~anchored)
    i, j = pi[rows]-1, pj[rows]-1
    ri, rj = radius[i], radius[j]
    delta = ri-rj
    diff = direction[i]-direction[j]
    norm = np.linalg.norm(diff, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_a = np.abs(delta)+2*np.log(-np.expm1(-np.abs(delta)))-np.log(4)
        log_sinh_i = ri + np.log(-np.expm1(-2*ri))-np.log(2)
        log_sinh_j = rj + np.log(-np.expm1(-2*rj))-np.log(2)
        log_b = log_sinh_i+log_sinh_j+2*np.log(norm)-np.log(4)
        log_c = np.logaddexp(log_a, log_b)
        weight_a, weight_b = np.exp(log_a-log_c), np.exp(log_b-log_c)
    distance = 2*np.logaddexp(.5*log_c, .5*np.logaddexp(log_c, 0))
    root = np.exp(.5*(log_c-np.logaddexp(log_c, 0)))
    radial_a = np.divide(weight_a, np.tanh(delta/2), out=np.zeros_like(delta), where=delta != 0)
    dr_i = root*(radial_a+weight_b/np.tanh(ri))
    dr_j = root*(-radial_a+weight_b/np.tanh(rj))
    predicted[rows] = distance/scale
    jacobian[rows, i*dimension] = dr_i*ri/scale
    jacobian[rows, j*dimension] = dr_j*rj/scale
    angular = np.divide(2*root*weight_b, norm**2, out=np.zeros_like(norm), where=norm>0)/scale
    for axis in range(dimension-1):
        jacobian[rows, i*dimension+axis+1] = angular*np.einsum('ij,ij->i', diff, derivatives[i,axis])
        jacobian[rows, j*dimension+axis+1] = -angular*np.einsum('ij,ij->i', diff, derivatives[j,axis])
    jacobian[rows, -1] = (dr_i*ri+dr_j*rj-distance)/scale
    return predicted[~anchored], jacobian[~anchored], radius, direction


def polar_objective(x, n, dimension, observed):
    predicted, jac, _, _ = polar_distances(x,n,dimension)
    residual = predicted-observed
    mse = np.mean(residual**2)
    if mse <= 0 or not np.isfinite(mse):
        raise FloatingPointError('Undefined shared variance')
    return .5*(np.log(2*np.pi*mse)+1), jac.T@residual/(len(residual)*mse)


def refine_polar(x, n, dimension, observed, max_blocks=20, iterations_per_block=1000,
                 gradient_tolerance=1e-6, label='', fixed_lambda=None):
    x = x.copy()
    if fixed_lambda is not None:
        x[-1] = np.log(fixed_lambda)
    active = np.arange(len(x)-int(fixed_lambda is not None))
    history = []
    stop = 'block budget exhausted'
    for block in range(max_blocks):
        predicted, jac, _, _ = polar_distances(x,n,dimension)
        residual = predicted-observed
        mse = np.mean(residual**2)
        value, gradient = polar_objective(x,n,dimension,observed)
        if np.max(np.abs(gradient[active])) <= gradient_tolerance:
            stop = 'original gradient criterion met'
            break
        fisher = jac.T@jac/(len(residual)*mse)
        units = 1/np.sqrt(np.maximum(np.diag(fisher), 1e-20))
        parameters = x[:-1].reshape(n,dimension)
        rotations = []
        field = np.zeros((n,dimension))
        field[:,1] = 1
        rotations.append(np.r_[field.ravel(),0]/units)
        if dimension == 3:
            theta,phi=parameters[:,1],parameters[:,2]
            for angular,polar in ((-np.cos(theta)/np.tan(phi),-np.sin(theta)),
                                   (-np.sin(theta)/np.tan(phi),np.cos(theta))):
                field=np.column_stack([np.zeros(n),angular,polar])
                rotations.append(np.r_[field.ravel(),0]/units)
        # Remove the translation/boost gauge too; no observed point is pinned.
        radius=np.exp(x[-1]+parameters[:,0])
        theta=parameters[:,1]
        if dimension == 2:
            for radial,angular in ((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta))):
                field=np.column_stack([radial/radius,angular/np.tanh(radius)])
                rotations.append(np.r_[field.ravel(),0]/units)
        else:
            phi=parameters[:,2]
            directions=np.column_stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
            dt=np.column_stack([-np.sin(theta)/np.sin(phi),np.cos(theta)/np.sin(phi),np.zeros(n)])
            dp=np.column_stack([np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),-np.sin(phi)])
            for axis in range(3):
                field=np.column_stack([directions[:,axis]/radius,dt[:,axis]/np.tanh(radius),dp[:,axis]/np.tanh(radius)])
                rotations.append(np.r_[field.ravel(),0]/units)
        quotient=null_space(np.asarray(rotations)[:,active])
        local_jac=(jac[:,active]*units[active])@quotient/np.sqrt(len(residual)*mse)
        eigenvalues, eigenvectors = np.linalg.eigh(local_jac.T@local_jac)
        ridge = max(eigenvalues[-1]*1e-9, 1e-12)
        transform = np.zeros((len(x),quotient.shape[1]))
        transform[active] = units[active,None]*(quotient@(eigenvectors/np.sqrt(np.maximum(eigenvalues,0)+ridge)))
        def objective(z):
            try:
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    loss,g = polar_objective(x+transform@z,n,dimension,observed)
                if np.isfinite(loss) and np.isfinite(g).all():
                    return loss,transform.T@g
            except FloatingPointError:
                pass
            return np.inf,np.zeros_like(z)
        result = minimize(objective,np.zeros(transform.shape[1]),jac=True,method='L-BFGS-B',
                          options=dict(maxiter=iterations_per_block,maxfun=iterations_per_block*50,
                                       ftol=1e-14,gtol=1e-9,maxls=50,maxcor=30))
        candidate=x+transform@result.x
        next_value,_=polar_objective(candidate,n,dimension,observed)
        if np.isfinite(next_value) and next_value<=value:
            x=candidate
        loss,g=polar_objective(x,n,dimension,observed)
        history.append(dict(block=block+1,mean_nll=loss,lambda_value=np.exp(x[-1]),
                            gradient_inf=np.max(np.abs(g[active])),loss_reduction=value-loss,
                            optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit)))
        print(f'{label} block={block+1}: loss={loss:.8f}, lambda={np.exp(x[-1]):.5g}, gradient={np.max(np.abs(g[active])):.3g}',flush=True)
        if np.max(np.abs(g[active]))<=gradient_tolerance:
            stop='original gradient criterion met'
            break
        if len(history)>=3 and all(row['loss_reduction']<1e-12 for row in history[-3:]):
            stop='objective stagnated; gradient criterion not met'
            break
    predicted,jac,radius,direction=polar_distances(x,n,dimension)
    residual=predicted-observed
    mse=np.mean(residual**2)
    loss,g=polar_objective(x,n,dimension,observed)
    # Stable conversion for visualization only. Fitting remains in radial/angular coordinates.
    exponential=np.exp(-2*radius)
    denominator=(1-direction[:,-1])+exponential*(1+direction[:,-1])
    chart=np.column_stack([(1-exponential[:,None])*direction[:,:-1]/denominator[:,None],
                           np.log(2)-radius-np.log(denominator)])
    return dict(parameters_vector=x,parameterization='polar',halfspace_coordinates=chart,residual=residual,
                history=pd.DataFrame(history),diagnostics=dict(dimension=dimension,mean_nll=loss,
                    lambda_value=float(np.exp(x[-1])),relative_rmse=float(np.linalg.norm(residual)/np.linalg.norm(observed)),
                    shared_sigma_scaled=float(np.sqrt(mse/2)),pair_residual_sd_scaled=float(np.sqrt(mse)),
                    gradient_inf=float(np.max(np.abs(g[active]))),converged=bool(np.max(np.abs(g[active]))<=gradient_tolerance),
                    lambda_fitted=fixed_lambda is None, fixed_lambda=fixed_lambda, lambda_score=float(g[-1]),
                    stop_reason=stop,maximum_hyperbolic_radius=float(radius.max())))


def polar_gradient_check(fit,n,dimension,observed):
    x=fit['parameters_vector']
    _,gradient=polar_objective(x,n,dimension,observed)
    prediction=polar_distances(x,n,dimension)[0]
    residual=prediction-observed
    mse=np.mean(residual**2)
    def loss_change(values):
        delta=polar_distances(values,n,dimension)[0]-prediction
        return .5*np.log1p(np.mean(delta*(2*residual+delta))/mse)
    rows=[]
    for k in range(len(x)):
        for step in (1e-4,1e-5,1e-6,1e-7,1e-8,1e-9):
            delta=step*max(1,abs(x[k]))
            plus,minus=x.copy(),x.copy()
            plus[k]+=delta;minus[k]-=delta
            numeric=(loss_change(plus)-loss_change(minus))/(plus[k]-minus[k])
            ratio=abs(numeric-gradient[k])/(1e-7+1e-3*max(abs(numeric),abs(gradient[k])))
            rows.append(dict(parameter=k,relative_step=step,analytic=gradient[k],numerical=numeric,
                             error_over_tolerance=ratio,agrees=bool(ratio<=1)))
    steps=pd.DataFrame(rows)
    checks=steps.groupby('parameter').agrees.apply(lambda v:bool(np.any(v.to_numpy()[:-1]&v.to_numpy()[1:])))
    return steps,checks

import jax.numpy as jnp
import jax

def loss_logistic(params, X, Y, changepoints, A):
    return jnp.mean((model_logistic(params,X,changepoints,A)-Y)**2)

def gamma(params, changepoints):
    k = params["k"]#[1]
    m = params["m"]#[2]
    delta = params["delta"]#[3:3+changepoints.shape[0]]
    
    gammas = jnp.zeros((delta.shape[0], 1))
    nom = jnp.concatenate((jnp.array([1e-8]), jnp.cumsum(delta)), 0)
    for s in range(delta.shape[0]):
        g = (changepoints[s] - m - jnp.sum(gammas[:s,:])) * (1 - (k + nom[s])/(k + nom[s+1]))
        gammas = jax.ops.index_update(gammas, jax.ops.index[s], g)
    return gammas

def model_logistic(params, X, changepoints, A):
    C = params["C"]#[0]
    k = params["k"]#[1]
    m = params["m"]#[2]
    delta = params["delta"]#[3:3+changepoints.shape[0]]
    beta = params["beta"]#[3+changepoints.shape[0]:]

    gammas = gamma(params, changepoints)
    T = jnp.expand_dims(X[:,-1], 1)
    trend = C/(1 + jnp.exp(-(k + jnp.dot(A,delta))*(T - m - jnp.dot(A,gammas))))
    seasonality = jnp.dot(X,beta)
    return trend+seasonality

@jax.jit
def update_logistic(X, Y, params, lr, changepoints, A):
    loss, grads = jax.value_and_grad(loss_logistic)(params,X,Y,changepoints,A)
    new_params = jax.tree_multimap(lambda p, g: p - lr*g, params, grads)
    return new_params, loss

def fit_logistic_prophet(X, Y, lr, epochs, changepoints, A):
    params = {}
    params["C"] = jnp.ones((1,)) * 5
    params["k"] = jnp.ones((1,)) * 0.1
    params["m"] = jnp.ones((1,)) * 0.1
    params["delta"] = jnp.ones((A.shape[1],1)) * 0.1
    params["beta"] = jnp.ones((X.shape[1],1)) * 0.1
    #params = jnp.ones((X.shape[1] + 3 + A.shape[1], 1)) * 0.1
    #params = jax.ops.index_update(params, jax.ops.index[0], 5)
    #params = jax.ops.index_update(params, jax.ops.index[2], 30)

    for i in range(epochs):
        params, loss = update_logistic(X,Y,params,lr,changepoints,A)
    print(loss)
    return params

def infer_logistic_prophet(X, params, changepoints, A):
    return model_logistic(params,X,changepoints,A)
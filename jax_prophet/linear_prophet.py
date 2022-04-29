import jax.numpy as jnp
import jax

def loss_fn(w, X, Y):
    return jnp.mean((jnp.dot(X,w)-Y)**2)

def init(X):
    return jnp.ones((X.shape[1], 1)) * 0.1

@jax.jit
def update(X, Y, w, lr):
    loss, grads = jax.value_and_grad(loss_fn)(w,X,Y)
    new_w = w - lr*grads
    return new_w, loss

def fit_linear_prophet(X, Y, lr, epochs):
    w = init(X)
    for _ in range(epochs):
        w, loss = update(X,Y,w,lr)
    print(loss)
    return w

def infer_linear_prophet(X, w):
    return jnp.dot(X,w)
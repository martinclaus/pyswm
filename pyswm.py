"""Simple shallow water model
System of PDE to solve:
u_t = -gη_x + fv
v_t = -gη_y - fu
η_t = -(Hu)_x - (Hv)_y

Coriolis parameter f may be constant, linearly varying
or prop. to sin(θ).

Grid: Curvilinear Arakawa C-grid
Time integration scheme: Heaps (1972)
"""

import numpy as np

nx = 100
ny = 50


def cyclic_index(ind, ax_len):
    """Cyclic index wrapping.

    Assumes negative indices to be larger than -ax_len
    """
    return (ind + ax_len) % ax_len


def cx(i): return cyclic_index(i, nx)


def cy(j): return cyclic_index(j, ny)


def create_var(shape=(ny, nx), dtype=np.float64):
    """Allocate empty numpy array."""
    return np.empty(shape, dtype=dtype)


def init_var(val=0., *args, **kwargs):
    """Initialize new variable with val."""
    return val * create_var(*args, **kwargs)


# initialize variables
u_t, v_t, η_t, u_n, v_n, η_n, u_np1, v_np1, η_np1 = [
    init_var() for _ in range(9)
]


def zonal_pressure_gradient(η, g, Δx):
    """Compute zonal pressure gradient.

    Returns -gη_x using centred differences and cyclic boundary conditions.
    """
    res = create_var(η.shape)
    for j in range(η.shape[-2]):
        for i in range(η.shape[-1]):
            res[j, i] = -g * (η[j, i] - η[j, cx(i - 1)]) / Δx[j, i]
    return res


def meridional_pressure_gradient(η, g, Δy):
    """Compute meridional pressure gradient.

    Returns -gη_y using centred differences and cyclic boundary conditions.
    """
    res = create_var(η.shape)
    for j in range(η.shape[-2]):
        for i in range(η.shape[-1]):
            res[j, i] = -g * (η[j, i] - η[cy(j), i]) / Δy[j, i]
    return res


def zonal_coriolis(v, f):
    """Compute Coriolis term in zonal momentum equation.

    Returns +fv using a four point average of v.
    """
    res = create_var(v.shape)
    for j in range(v.shape[-2]):
        for i in range(v.shape[-1]):
            res[j, i] = f * .25 * (
                v[j, i]
                + v[cy(j + 1), i]
                + v[cy(j + 1), cx(i - 1)]
                + v[j, cx(i - 1)]
            )
    return res


def meridional_coriolis(u, f):
    """Compute Coriolis term in zonal momentum equation.

    Returns -fu using a four point average of v.
    """
    res = create_var(u.shape)
    for j in range(u.shape[-2]):
        for i in range(u.shape[-1]):
            res[j, i] = (-1.) * f * .25 * (
                u[j, i]
                + u[cy(j - 1), i]
                + u[cy(j - 1), cx(i + 1)]
                + u[j, cx(i + 1)]
            )
    return res


def zonal_convergence(u, H, Δx, Δy, Δy_u):
    """Compute convergence of zonal flow.

    Returns -(Hu)_x taking account of the curvature of the grid.
    """
    res = create_var(u.shape)
    for j in range(u.shape[-2]):
        for i in range(u.shape[-1]):
            res[j, i] = (-1) * (
                H[j, cx(i + 1)] * u[j, cx(i + 1)] * Δy_u[j, cx(i + 1)]
                - H[j, i] * u[j, i] * Δy_u[j, i]
            ) / (Δx[j, i] * Δy[j, i])
    return res


def meridional_convergence(v, H, Δx, Δy, Δx_v):
    """Compute convergence of meridional flow.

    Returns -(Hv)_y taking account of the curvature of the grid.
    """
    res = create_var(v.shape)
    for j in range(v.shape[-2]):
        for i in range(v.shape[-1]):
            res[j, i] = (-1) * (
                H[cy(j + 1), i] * v[cy(j + 1), i] * Δx_v[cy(j + 1), i]
                - H[j, i] * v[j, i] * Δx_v[j, i]
            ) / (Δx[j, i] * Δy[j, i])
    return res


def compute_tendency_u(v, η, g, f, Δx):
    """Compute sum of right hand side terms of the zonal momentum equation."""
    res = init_var(0., v.shape)
    res += zonal_pressure_gradient(η, g, Δx)
    res += zonal_coriolis(v, f)
    return res


def compute_tendency_v(u, η, g, f, Δy):
    """Compute sum of right hand side terms of the zonal momentum equation."""
    res = init_var(0., u.shape)
    res += meridional_pressure_gradient(η, g, Δy)
    res += meridional_coriolis(u, f)
    return res


def compute_tendency_η(u, v, H, Δx, Δy, Δy_u, Δx_v):
    """Compute sum of right hand side terms of the continuity equation."""
    res = init_var(0., u.shape)
    res += zonal_convergence(u, H, Δx, Δy, Δy_u)
    res += meridional_convergence(v, H, Δx, Δy, Δx_v)
    return res


def integrate_FW(var, g_var, Δt):
    """Compute state at next time step using Euler Forward."""
    var_next = create_var(var.shape)
    var_next[...] = var + Δt * g_var
    return var_next


def integrate_heaps(u, v, η, H, f, g, Δt, Δx_η, Δy_η, Δx_u, Δy_u, Δx_v, Δy_v):
    """Compute state at next time step using Heaps (1972).

    η_(n+1) = η_n + Δt * G_η(u_n, v_n)
    u_(n+1) = u_n + Δt * G_u(η_(n+1), v_n)
    v_(n+1) = v_n + Δt * G_v(η_(n+1), u_(n+1))

    returns η_(n+1), u_(n+1), v_(n+1)
    """
    # η_next, u_next, v_next = (
    #     init_var(0., η.shape), init_var(0., u.shape), init_var(0., v.shape)
    # )
    η_next = integrate_FW(
        η,
        compute_tendency_η(u, v, H, Δx_η, Δy_η, Δy_u, Δx_v),
        Δt
    )
    u_next = integrate_FW(
        u,
        compute_tendency_u(v, η_next, g, f, Δx_u),
        Δt
    )
    v_next = integrate_FW(
        u,
        compute_tendency_v(u, η, g, f, Δy_v),
        Δt
    )
    return η_next, u_next, v_next

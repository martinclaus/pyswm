"""Simple shallow water model
System of PDE to solve:
u_t = -gη_x + fv
v_t = -gη_y - fu
η_t = -(hu)_x - (hv)_y

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
    buffer = np.empty(shape, dtype=dtype)
    return buffer


def init_var(val=0., *args, **kwargs):
    """Initialize new variable with val."""
    var = create_var(*args, **kwargs)
    var[...] = val
    return var


def zonal_pressure_gradient(eta, g, dx, ocean_eta):
    """Compute zonal pressure gradient.

    Returns -gη_x using centred differences and cyclic boundary conditions.
    """
    res = create_var(eta.shape)
    for j in range(eta.shape[-2]):
        for i in range(eta.shape[-1]):
            res[j, i] = (
                -g 
                * (
                    eta[j, i] * ocean_eta[j, i]
                    - eta[j, cx(i - 1)] * ocean_eta[j, cx(i - 1)]
                )
                / dx[j, i]
            )
    return res


def meridional_pressure_gradient(eta, g, dy, ocean_eta):
    """Compute meridional pressure gradient.

    Returns -gη_y using centred differences and cyclic boundary conditions.
    """
    res = create_var(eta.shape)
    for j in range(eta.shape[-2]):
        for i in range(eta.shape[-1]):
            res[j, i] = (
                -g
                * (
                    eta[j, i] * ocean_eta[j, i]
                    - eta[cy(j - 1), i] * ocean_eta[cy(j - 1), i]
                )
                / dy[j, i])
    return res


def zonal_coriolis(v, f, ocean_v):
    """Compute Coriolis term in zonal momentum equation.

    Returns +fv using a four point average of v.
    """
    res = create_var(v.shape)
    for j in range(v.shape[-2]):
        for i in range(v.shape[-1]):
            res[j, i] = f[j, i] * .25 * (
                v[j, i] * ocean_v[j, i]
                + v[cy(j + 1), i] * ocean_v[cy(j + 1), i]
                + v[cy(j + 1), cx(i - 1)] * ocean_v[cy(j + 1), cx(i - 1)]
                + v[j, cx(i - 1)] * ocean_v[j, cx(i - 1)]
            )
    return res


def meridional_coriolis(u, f, ocean_u):
    """Compute Coriolis term in zonal momentum equation.

    Returns -fu using a four point average of u.
    """
    res = create_var(u.shape)
    for j in range(u.shape[-2]):
        for i in range(u.shape[-1]):
            res[j, i] = (-1.) * f[j, i] * .25 * (
                u[j, i] * ocean_u[j, i]
                + u[cy(j - 1), i] * ocean_u[cy(j - 1), i]
                + u[cy(j - 1), cx(i + 1)] * ocean_u[cy(j - 1), cx(i + 1)]
                + u[j, cx(i + 1)] * ocean_u[j, cx(i + 1)]
            )
    return res


def zonal_convergence(u, h, dx, dy, dy_u, ocean_u):
    """Compute convergence of zonal flow.

    Returns -(hu)_x taking account of the curvature of the grid.
    """
    res = create_var(u.shape)
    for j in range(u.shape[-2]):
        for i in range(u.shape[-1]):
            res[j, i] = (-1) * (
                h[j, cx(i + 1)] * u[j, cx(i + 1)] * dy_u[j, cx(i + 1)] * ocean_u[j, cx(i + 1)]
                - h[j, i] * u[j, i] * dy_u[j, i] * ocean_u[j, i]
            ) / (dx[j, i] * dy[j, i])
    return res


def meridional_convergence(v, h, dx, dy, dx_v, ocean_v):
    """Compute convergence of meridional flow.

    Returns -(hv)_y taking account of the curvature of the grid.
    """
    res = create_var(v.shape)
    for j in range(v.shape[-2]):
        for i in range(v.shape[-1]):
            res[j, i] = (-1) * (
                h[cy(j + 1), i] * v[cy(j + 1), i] * dx_v[cy(j + 1), i] * ocean_v[cy(j + 1), i]
                - h[j, i] * v[j, i] * dx_v[j, i] * ocean_v[j, i]
            ) / (dx[j, i] * dy[j, i])
    return res


def compute_tendency_u(v, eta, g, f, dx):
    """Compute sum of right hand side terms of the zonal momentum equation."""
    res = init_var(0., v.shape)
    res += zonal_pressure_gradient(eta, g, dx)
    res += zonal_coriolis(v, f)
    return res


def compute_tendency_v(u, eta, g, f, dy):
    """Compute sum of right hand side terms of the zonal momentum equation."""
    res = init_var(0., u.shape)
    res += meridional_pressure_gradient(eta, g, dy)
    res += meridional_coriolis(u, f)
    return res


def compute_tendency_eta(u, v, h_u, h_v, dx, dy, dy_u, dx_v):
    """Compute sum of right hand side terms of the continuity equation."""
    res = init_var(0., u.shape)
    res += zonal_convergence(u, h_u, dx, dy, dy_u)
    res += meridional_convergence(v, h_v, dx, dy, dx_v)
    return res


def integrate_FW(var, g_var, dt):
    """Compute state at next time step using Euler Forward."""
    var_next = create_var(var.shape)
    var_next[...] = var + dt * g_var
    return var_next


def integrate_heaps(
    u, v, eta,
    h_u, h_v, f, g,
    dt, dx_eta, dy_eta, dx_u, dy_u, dx_v, dy_v
):
    """Compute state at next time step using Heaps (1972).

    η_(n+1) = η_n + dt * G_η(u_n, v_n)
    u_(n+1) = u_n + dt * G_u(η_(n+1), v_n)
    v_(n+1) = v_n + dt * G_v(η_(n+1), u_(n+1))

    returns η_(n+1), u_(n+1), v_(n+1)
    """
    # eta_next, u_next, v_next = (
    #     init_var(0., eta.shape), init_var(0., u.shape), init_var(0., v.shape)
    # )
    eta_next = integrate_FW(
        eta,
        compute_tendency_eta(u, v, h_u, h_v, dx_eta, dy_eta, dy_u, dx_v),
        dt
    )
    u_next = integrate_FW(
        u,
        compute_tendency_u(v, eta_next, g, f, dx_u),
        dt
    )
    v_next = integrate_FW(
        u,
        compute_tendency_v(u, eta, g, f, dy_v),
        dt
    )
    return eta_next, u_next, v_next

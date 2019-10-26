import numpy as np


if __name__ == "__main__":

    L = 1  # length of the medium
    Nx = 10  # number of nodes (layers) in the medium
    T = 10  # upper time limit
    Nt = 100  # number of nodes in the time limit
    a = 500  # conductivity coefficient

    x = np.linspace(0, L, Nx + 1)  # mesh points in space
    dx = x[1] - x[0]
    t = linspace(0, T, Nt + 1)  # mesh points in time
    dt = t[1] - t[0]
    F = a * dt / dx ** 2
    u = zeros(Nx + 1)  # unknown u at new time level
    u_1 = zeros(Nx + 1)  # u at the previous time level

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx + 1):
        u_1[i] = I(x[i])

    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[i] = u_1[i] + F * (u_1[i - 1] - 2 * u_1[i] + u_1[i + 1])

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0

        # Update u_1 before next step
        u_1[:] = u

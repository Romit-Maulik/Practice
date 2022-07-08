# Romit Maulik - PhD Student - Computational Fluid Dynamics Laboratory
# email:romit.maulik@okstate.edu
# 10-2-2017
# Import statements - these are libraries
import numpy as np  # This imports array data structures
import matplotlib.pyplot as plt  # This imports plot tools

def initialize(array):  # Initializes in fourier domain
    global nx, nr
    pi = np.pi

    dx = xlength / float(nx)
    dx1 = dx / float(nr)

    nx_new = nx * nr
    xlength_new = dx1 * float(nx_new)

    kx = np.array([(2 * pi) * i / xlength_new for i in
                   list(range(0, nx_new // 2)) + [0] + list(range(-nx_new // 2 + 1, 0))])  # Note no im heref
    acons = 2.0 / (10.0 ** (5.0)) / (3.0 * (pi ** (0.5)))

    array_hat = np.zeros(nx_new, dtype=np.complex)  # The array is of type complex
    phase = np.zeros(2 * nx_new, dtype='double')

    np.random.seed(0)
    rand = np.random.uniform(0.0, 1.0)
    phase[0] = np.cos(2.0 * pi * rand)
    phase[1] = 0.0
    phase[nx_new] = np.cos(2.0 * pi * rand)
    phase[nx_new + 1] = 0.0

    k = 3
    for i in range(1, nx_new // 2):
        rand = np.random.uniform(0.0, 1.0)
        phase[k - 1] = np.cos(2.0 * pi * rand)
        phase[k] = np.sin(2.0 * pi * rand)
        phase[2 * nx_new - k + 1] = np.cos(2.0 * pi * rand)
        phase[2 * nx_new - k + 2] = -np.sin(2.0 * pi * rand)
        k = k + 2

    k = 0
    for i in range(0, nx_new):
        espec_ip = np.exp(-(kx[i] / 10.0) ** (2.0))
        espec = acons * (kx[i] ** 4) * espec_ip
        array_hat[i] = nx_new * (np.sqrt(2.0 * espec) * (phase[k] + phase[k + 1]))
        k = k + 2

    temp_array = np.real(np.fft.ifft(array_hat))
    copy_array = np.zeros(nx, dtype='double')

    for i in range(0, nx):
        copy_array[i] = temp_array[i * nr]

    np.copyto(array, np.fft.fft(copy_array))
    del temp_array, copy_array


def spectra_calculation(array_hat):
    # Normalizing data
    global nx
    array_new = np.copy(array_hat / float(nx))
    # Energy Spectrum
    espec = 0.5 * np.absolute(array_new)**2
    # Angle Averaging
    eplot = np.zeros(nx // 2, dtype='double')
    for i in range(1, nx // 2):
        eplot[i] = 0.5 * (espec[i] + espec[nx - i])

    return eplot


def rkcn_spectral_1D(array_hat, time):
    global alpha, nx, num_op, xlength
    pi = np.pi

    kx_linear = np.array([(2 * pi) * i / xlength for i in list(range(0, nx // 2)) + [0] + list(range(-nx // 2 + 1, 0))])
    kx_nonlinear = np.array([(2 * pi) * i / xlength for i in list(range(0, nx // 2)) + [0] + list(range(-nx // 2 + 1, 0))])
    k2 = kx_linear ** 2
    # Two Third Rule for nonlinear term
    ll = ((nx // 2) - 1) // 6 * nx
    ul = ((nx // 2) + 1) // 6 * nx
    for i in range(ll, ul):
        kx_nonlinear[i] = 0.0

    h = xlength / nx
    x = [h * i for i in range(1, nx + 1)]

    array_hat_0 = array_hat

    dt = 1.0e-5
    t = 0.0
    op = 1.0
    while t < time:
        t = t + dt
        # TVDRK Stage 1
        du_hat = complex(0,1)*kx_nonlinear*array_hat_0
        du = np.fft.ifft(du_hat)
        udu= np.fft.ifft(array_hat_0)*du
        nonlinear_term_0 = np.fft.fft(-udu)

        premult = 1.0 / (1.0 + 4.0 / 15.0 * dt * alpha * k2 )
        array_hat_1 = premult * (
        array_hat_0 + 8.0 / 15 * dt * nonlinear_term_0 - 8.0 / 15.0 * dt * alpha * k2 / 2.0 * array_hat_0)

        # TVDRK Stage 2
        du_hat = complex(0,1)*kx_nonlinear * array_hat_1
        du = np.fft.ifft(du_hat)
        udu = np.fft.ifft(array_hat_1) * du
        nonlinear_term_1 = np.fft.fft(-udu)

        premult = 1.0 / (1.0 + 1.0 / 15.0 * dt * alpha * k2 )
        array_hat_2 = premult * (
            array_hat_1 + 5.0 / 12.0 * dt * nonlinear_term_1 - 17.0 / 60.0 * dt * nonlinear_term_0 - 2.0 / 15.0 * dt * alpha * k2 / 2.0 * array_hat_1)

        # TVDRK Stage 3
        du_hat = complex(0,1)*kx_nonlinear * array_hat_2
        du = np.fft.ifft(du_hat)
        udu = np.fft.ifft(array_hat_2) * du
        nonlinear_term_2 = np.fft.fft(-udu)

        premult = 1.0 / (1.0 + 1.0 / 6.0 * dt * alpha * k2 )
        array_hat_0 = premult * (
            array_hat_2 + 3.0 / 4.0 * dt * nonlinear_term_2 - 5.0 / 12.0 * dt * nonlinear_term_1 - 1.0 / 3.0 * dt * alpha * k2 / 2.0 * array_hat_2)

        del nonlinear_term_1, nonlinear_term_0, nonlinear_term_2, array_hat_1, array_hat_2

        if t/time > op/num_op:
            #Plotting Field
            plt.figure(1)
            plt.plot(x, np.real(np.fft.ifft(array_hat_0)), label='Time '+str(t))

            # Plotting spectra
            plt.figure(2)
            kx_plot = np.array([float(i) for i in range(0, nx // 2)])
            spec = spectra_calculation(array_hat_0)
            plt.plot(kx_plot, spec, label='Time '+str(t))
            del kx_plot
            op = op+1

    plt.figure(2)
    # Plotting reference (k^-2)
    kref = np.array([4, 100])
    eref = np.array([0.1, 16 * 1.0e-5])
    plt.plot(kref, eref, '--', label='Reference')
    plt.title('1D - Viscous Burgers Equation, Spectra')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.yscale('log')
    plt.xscale('log')

    plt.figure(1)
    plt.title('1D - Viscous Burgers Equation, Field')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.show()


# Main body of execution
if __name__ == "__main__":
    nx = 2048
    nr = int(32768 / nx)
    xlength = 2.0 * np.pi
    alpha = 5.0e-3  # Viscosity coefficient
    num_op = 5  # Number of outputs
    u = np.zeros(nx, dtype='complex')  # The array is of type complex
    initialize(u)

    # Plotting initial fields
    h = xlength / nx
    x = [h * i for i in range(1, nx + 1)]
    plt.figure()
    plt.plot(x, np.real(np.fft.ifft(u)))
    plt.title('Initial conditions')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.show()

    rkcn_spectral_1D(u, 0.2)

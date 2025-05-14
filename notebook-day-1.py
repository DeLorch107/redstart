import marimo

__generated_with = "0.13.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FuncAnimation, PillowWriter, np, plt, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell
def _(FuncAnimation, PillowWriter, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6))
        num_frames = 100
        fps = 30 # Number of frames per second
    
        def animate(frame_index):
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
        
            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()
        
            if hasattr(make_video, 'pbar'):
                make_video.pbar.update(1)
    
        make_video.pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
    
        # Use PillowWriter instead of FFMpegWriter (no external dependencies)
        writer = PillowWriter(fps=fps)
    
        # Change output filename to .gif since we're using PillowWriter
        if output.endswith('.mp4'):
            output = output.replace('.mp4', '.gif')
    
        anim.save(output, writer=writer)
    
        print()
        print(f"Animation saved as {output!r}")
        return output

    _filename = "wave_animation.gif"  # Changed to .gif
    output_file = make_video(_filename)

    # Use marimo's image display
    mo.image(output_file)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell
def _():
    g = 1.0  # gravity constant in m/s^2
    M = 1.0  # mass in kg
    l = 1.0  # half-length of the booster in meters (since total length is 2 meters)
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Since the force is at angle \( \varphi \) from the booster axis, and the booster axis is at angle \( \theta \) from vertical,  
    the force makes an angle of \( (\theta + \varphi) \) with the vertical, measured counterclockwise.

    Therefore:

    \[
    f_x = -f \cdot \sin(\theta + \varphi)
    \]

    \[
    f_y = f \cdot \cos(\theta + \varphi)
    \]

    """
    )
    return


@app.cell
def _(np):
    f = 0
    angle_theta = 0
    angle_phi = 0
    f_x = -f * np.sin(angle_phi+angle_theta)
    f_y = f * np.cos(angle_phi+angle_theta)
    return angle_phi, angle_theta, f


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Given:
    1.  Gravity: $(0, -Mg)$.
    2.  Reactor force: $(f_x, f_y)$ where:
   
        $f_x = -f \sin(\theta+\phi)$
    
        $f_y = f \cos(\theta+\phi)$

    We apply Newton's Second Law ($\vec{F} = M\vec{a}$) for the $x$ and $y$ components.

    **For the x-coordinate:**

    The sum of forces in the x-direction is $\sum F_x$.
    The only force component in the x-direction is the reactor force $f_x$.
    The gravitational force has no x-component.

    $M\ddot{x} = f_x$

    Substituting the definition for $f_x$:

    $M\ddot{x} = -f \sin(\theta+\phi)$

    So, the equation for the x-acceleration is:

    $\ddot{x} = -\frac{f}{M} \sin(\theta+\phi)$

    **For the y-coordinate:**

    The sum of forces in the y-direction is $\sum F_y$.
    The forces in the y-direction are the reactor force $f_y$ and the gravitational force $-Mg$.

    $M\ddot{y} = f_y - Mg$

    Substituting the definition for $f_y$:

    $M\ddot{y} = f \cos(\theta+\phi) - Mg$

    So, the equation for the y-acceleration is:

    $\ddot{y} = \frac{f}{M} \cos(\theta+\phi) - g$

    **Therefore, with this latest set of definitions for $f_x$ and $f_y$, the ordinary differential equations governing the motion of the center of mass $(x,y)$ are:**

    $\ddot{x} = -\frac{f}{M} \sin(\theta+\phi)$

    $\ddot{y} = \frac{f}{M} \cos(\theta+\phi) - g$
    """
    )
    return


@app.cell
def _(M, angle_phi, angle_theta, f, g, np):
    # Declare/calculate ddot_x (x-acceleration)
    ddot_x = -(f / M) * np.sin(angle_phi+angle_theta)

    # Declare/calculate ddot_y (y-acceleration)
    ddot_y = (f / M) * np.cos(angle_phi+angle_theta) - g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Okay, here's the rewritten section keeping the variables as letters:

    ## Computing the Moment of Inertia of the Booster

    We need to calculate the moment of inertia $J$ of the booster about its center of mass.

    Given information:
    - The booster is a rigid tube of total length, let's call this $L_{total}$. The problem states this total length is $2\ell$.
    - Mass $M$ is uniformly distributed along its length.
    - The moment of inertia is calculated about the center of mass.

    For a uniform rod of mass $M_{rod}$ and length $L_{rod}$ rotating about its center of mass, the general formula for the moment of inertia is:

    $J_{rod} = \frac{1}{12}M_{rod}(L_{rod})^2$

    In our case, for the booster:
    - The mass of the booster is $M$.
    - The total length of the booster is $2\ell$.

    Substituting these specific parameters into the general formula:

    $J = \frac{1}{12}M(2\ell)^2$

    We can simplify this expression:

    $J = \frac{1}{12}M(4\ell^2)$
    $J = \frac{4}{12}M\ell^2$
    $J = \frac{1}{3}M\ell^2$

    Therefore, the moment of inertia of the booster is $J = \frac{1}{3}M\ell^2$.

    """
    )
    return


@app.cell
def _(M, l):
    J = 1/3 * M * l**2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ordinary Differential Equation for the Tilt Angle θ

    To determine the differential equation for the tilt angle θ, we need to analyze the rotational dynamics of the booster using the rotational form of Newton's Second Law:

    $J\ddot{\theta} = \tau_{total}$

    where:

    -   $J$ is the moment of inertia of the booster about its center of mass (as defined in the previous section, $J = \frac{1}{3}M\ell^2$).
    -   $\ddot{\theta}$ is the angular acceleration.
    -   $\tau_{total}$ is the total torque acting on the booster.

    The torques acting on the booster include:

    1.  **Torque due to the reactor force**:
        The reactor force $f$ acts at one end of the booster (a distance $\ell$ from the center of mass) at an angle $\phi$ relative to the booster's longitudinal axis. This creates a torque of:
        $\tau_{reactor} = - \ell \cdot f \cdot \sin(\phi)$

        The $\sin(\phi)$ term appears because only the component of force perpendicular to the lever arm (the booster's half-length $\ell$) contributes to the torque about the center of mass.

    2.  **Torque due to gravity**:
        For a uniform rod, gravity effectively acts at the center of mass. Therefore, the torque due to gravity about the center of mass is zero.
        $\tau_{gravity} = 0$

    Therefore, the total torque is solely due to the reactor force:
    $\tau_{total} = - \ell \cdot f \cdot \sin(\phi)$

    Substituting this into the rotational equation of motion (using the $J$ defined previously):
    $J\ddot{\theta} = - \ell \cdot f \cdot \sin(\phi)$

    To solve for $\ddot{\theta}$, we divide both sides by $J$:
    $\ddot{\theta} = - \frac{\ell \cdot f \cdot \sin(\phi)}{J}$

    Therefore, the ordinary differential equation governing the tilt angle $\theta$, expressed in terms of the previously defined moment of inertia $J$, the half-length $\ell$, the reactor force magnitude $f$, and the thrust angle $\phi$, is:
    $\ddot{\theta} = - \frac{\ell f \sin(\phi)}{J}$


    (If one were to then substitute the expression for $J = \frac{1}{3}M\ell^2$, it would simplify to $\ddot{\theta} = - \frac{3f \sin(\phi)}{M\ell}$ as derived before, but the request was to keep $J$ as the defined variable in this step.)
    """
    )
    return


@app.cell
def _(M, angle_phi, f, l, np):
    ddot_angle_theta = - (3 * f * np.sin(angle_phi)) / (M * l)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell
def _(g, l, np, plt):
    from scipy.integrate import solve_ivp

    def redstart_solve(t_span, y0, f_phi):
        """
        Solves the Redstart booster equations of motion.
    
        Parameters:
        -----------
        t_span : list or tuple
            Initial and final time [t0, tf]
        y0 : list or ndarray
            Initial state [x, dx, y, dy, theta, dtheta]
        f_phi : function
            Function that takes (t, y) and returns [f, phi]
        
        Returns:
        --------
        sol : function
            Function that takes a time t and returns the state
        """
        def dynamics(t, y):
            """
            The dynamics of the system.
        
            Parameters:
            -----------
            t : float
                Current time
            y : ndarray
                Current state [x, dx, y, dy, theta, dtheta]
            
            Returns:
            --------
            dydt : ndarray
                Time derivative of state [dx, ddx, dy, ddy, dtheta, ddtheta]
            """
            # Unpack state
            x, dx, y, dy, theta, dtheta = y
        
            # Get control inputs
            f, phi = f_phi(t, y)
        
            # Compute derivatives
            ddx = f * np.sin(theta + phi)
            ddy = -f * np.cos(theta + phi) - g
            ddtheta = - 3 * f * np.sin(phi)
        
            return np.array([dx, ddx, dy, ddy, dtheta, ddtheta])
    
        # Solve ODE
        ode_sol = solve_ivp(
            dynamics, 
            t_span, 
            y0, 
            method='RK45', 
            t_eval=None, 
            rtol=1e-6, 
            atol=1e-9,
            dense_output=True
        )
    
        # Create solution function that properly handles the interpolation
        def sol(t):
            if np.isscalar(t):
                # Handle single time point
                return ode_sol.sol(t)
            else:
                # Handle array of time points
                result = np.zeros((len(y0), len(t)))
                for i, ti in enumerate(t):
                    result[:, i] = ode_sol.sol(ti)
                return result
    
        return sol

    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # state: [x, dx, y, dy, theta, dtheta]
    
        def f_phi(t, y):
            return np.array([0.0, 0.0])  # input [f, phi]
    
        sol = redstart_solve(t_span, y0, f_phi)
    
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]  # Extract y-position component
    
        plt.figure(figsize=(10, 6))
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
    
        return plt.gcf()

    # Test the free fall example
    fig = free_fall_example()
    plt.show()
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


    $$
    y(0)=10,\quad \dot y(0)=0,\quad y(5)=\ell=1,\quad \dot y(5)=0
    $$

    and then computing the thrust profile $f(t)$ via

    $$
    \ddot y + g = -f(t)
    $$

    from your ODE $\ddot y = -f - g$.  A cubic

    $$
    y(t) = a t^3 + b t^2 + 10
    $$

    with

    $$
    \dot y(t) = 3a t^2 + 2b t
    $$

    gives four equations in $a,b$.  Solving,

    $$
    \begin{cases}
    125a +25b = -9,\\
    75a +10b = 0,
    \end{cases}
    \quad\Longrightarrow\quad
    a = 0.144,\; b = -1.08.
    $$

    Thus

    $$
    y(t)=0.144\,t^3-1.08\,t^2+10,\qquad
    \ddot y(t)=0.864\,t-2.16,
    $$

    and so

    $$
    f(t) = -\bigl(\ddot y(t)+g\bigr)
           = -\bigl(0.864\,t-2.16 +1.0\bigr)
           = -0.864\,t +1.16.
    $$

    Plug that into your solver (with $\phi=0$) and you’ll see
    $y(5)=1$ and $\dot y(5)=0$ exactly.  Here’s the full code to do it:


    What you’ll see:

    * A smooth descent from $y(0)=10$ down to $y(5)=1$.
    * $\dot y(5)$ printed as essentially zero (within solver tolerance).

    No sugar‑coating—it just works.

    """
    )
    return


@app.cell
def _(l, np, plt, redstart_solve):
    # Pre‐computed cubic coefficients
    a = 0.144
    b = -1.08


    def thrust_profile(t):
        # f(t) = -0.864 t + 1.16
        return -0.864*t + 1.16

    def free_boost_example():
        t_span = [0.0, 5.0]
        # Initial state: x, dx, y, dy, theta, dtheta
        y0 = [0, 0, 10, 0, 0, 0]

        def f_phi(t, y):
            return np.array([thrust_profile(t), 0.0])  # phi=0

        sol = redstart_solve(t_span, y0, f_phi)

        t = np.linspace(*t_span, 500)
        states = sol(t)
        y_t   = states[2]
        dy_t  = states[3]

        # Plot trajectory
        plt.figure(figsize=(8,5))
        plt.plot(t, y_t, label="y(t)")
        plt.plot(t, l*np.ones_like(t), 'k--', label="y = ℓ")
        plt.xlabel("t [s]")
        plt.ylabel("height y [m]")
        plt.title("Forced descent to y(5)=ℓ with y'(5)=0")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Print final values to check
        y5, dy5 = float(y_t[-1]), float(dy_t[-1])
        print(f"At t=5: y = {y5:.6f}, y_dot = {dy5:.6f}")

    # Run it
    free_boost_example()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


if __name__ == "__main__":
    app.run()

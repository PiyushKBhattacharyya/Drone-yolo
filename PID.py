class PID:
    def __init__(self, K_p: float = 0.4, K_d: float = 0.0, K_i: float = 0.0, dt: float = 0.5):
        """
        Initializes a PID controller with given parameters.

        Args:
            K_p (float): Proportional gain.
            K_d (float): Derivative gain.
            K_i (float): Integral gain.
            dt (float): Time step for the controller.
        """
        self.K_p = K_p  # Proportional gain
        self.K_d = K_d  # Derivative gain
        self.K_i = K_i  # Integral gain
        self.dt = dt  # Time step (interval)
        
        self.w = 0  # Desired velocity
        self.velocity = 0  # Current velocity
        self.errorsum = 0  # Cumulative error for the integral term
        self.actual_previous = 0  # Previous actual value for derivative calculation

    def step(self, desired: float, actual: float) -> float:
        """
        Computes the PID control output for the given desired and actual values.

        Args:
            desired (float): The desired setpoint value.
            actual (float): The current actual value.

        Returns:
            float: Control output (u).
        """
        # Proportional term
        error = desired - actual
        
        # Integral term
        self.errorsum += error * self.dt
        
        # Derivative term (rate of change of actual value)
        self.velocity = (actual - self.actual_previous) / self.dt
        
        # Compute control output
        u = (
            self.K_p * error
            + self.K_d * (self.w - self.velocity)
            + self.K_i * self.errorsum
        )
        
        # Update previous actual value for next step
        self.actual_previous = actual
        
        return u

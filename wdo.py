import numpy as np
import matplotlib.pyplot as plt

class WDOParams:
    def __init__(self):
        # Parameters from Image 1 for Sphere function
        self.alpha = 0.75        # α (air friction coefficient)
        self.max_velocity = 0.25 # Vmax
        self.gravity = 0.700     # g (gravitational constant)
        self.coriolis = -8       # c (coriolis effect)
        
        # Other parameters
        self.population_size = 100
        self.dimensions = 5      # N dimensions
        self.max_iterations = 1000
        self.RT = 3             # RT (temperature)
        self.archive_size = 100  # Size of external archive
        
        # Search space for Sphere function
        self.dim_min = -10      # xi ∈ [-10,10]
        self.dim_max = 10
        self.x_optimal = 2.0    # x* = (2,2,...,2)

class WindDrivenOptimization:
    def __init__(self, params):
        self.params = params
        # Step 1: Initialize parameters and population
        self.positions = 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        self.velocities = params.max_velocity * 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        
        # Initialize objectives and archive
        self.objectives = np.zeros((params.population_size, 2))
        self.archive = []
        self.pressure_history = []
        
    def evaluate_objectives(self, position):
        """Evaluate multiple objectives for Pareto optimization"""
        x = ((self.params.dim_max - self.params.dim_min) * (position + 1) / 2) + self.params.dim_min
        
        # Objective 1: Sphere function F_SPH(x) = Σ(xi - 2)^2
        f1 = np.sum(np.power((x - self.params.x_optimal), 2))
        
        # Objective 2: Spread/Smoothness of solution
        f2 = np.std(x)  # Standard deviation as second objective
        
        return np.array([f1, f2])

    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2"""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def non_dominating_sort(self):
        """Step 3: Perform non-dominating sort"""
        ranks = np.zeros(self.params.population_size)
        for i in range(self.params.population_size):
            dominated_count = 0
            for j in range(self.params.population_size):
                if i != j:
                    if self.dominates(self.objectives[j], self.objectives[i]):
                        dominated_count += 1
            ranks[i] = dominated_count + 1
        return ranks

    def update_archive(self):
        """Step 6: Update external archive"""
        # Combine current solutions with archive
        all_positions = list(self.positions)
        if self.archive:
            all_positions.extend(self.archive)
        
        all_objectives = [self.evaluate_objectives(pos) for pos in all_positions]
        
        # Find non-dominated solutions
        non_dominated_idx = []
        for i in range(len(all_positions)):
            dominated = False
            for j in range(len(all_positions)):
                if i != j and self.dominates(all_objectives[j], all_objectives[i]):
                    dominated = True
                    break
            if not dominated:
                non_dominated_idx.append(i)
        
        # Update archive with non-dominated solutions
        self.archive = [all_positions[i] for i in non_dominated_idx[:self.params.archive_size]]

    def optimize(self):
        for iteration in range(self.params.max_iterations):
            # Step 3: Evaluate objectives and perform non-dominating sort
            for i in range(self.params.population_size):
                self.objectives[i] = self.evaluate_objectives(self.positions[i])
            
            pareto_ranks = self.non_dominating_sort()
            
            # Step 4: Take Pareto-front rank into external population
            self.update_archive()
            
            # Step 5: Update velocities
            for i in range(self.params.population_size):
                random_perm = np.random.permutation(self.params.dimensions)
                velocity_other = self.velocities[i, random_perm]
                
                # Select random solution from archive as global best
                if self.archive:
                    global_best = self.archive[np.random.randint(len(self.archive))]
                else:
                    best_rank_idx = np.argmin(pareto_ranks)
                    global_best = self.positions[best_rank_idx]
                
                # Update velocity using rank information
                rank_factor = 1.0 / (pareto_ranks[i] + 1)
                self.velocities[i] = (
                    (1 - self.params.alpha) * self.velocities[i] +              # Friction
                    (-self.params.gravity * self.positions[i]) +                # Gravity
                    (rank_factor * (global_best - self.positions[i]) * self.params.RT) + # Pressure
                    (self.params.coriolis * velocity_other * rank_factor)       # Coriolis
                )
                
                # Limit velocity
                self.velocities[i] = np.clip(self.velocities[i], 
                                           -self.params.max_velocity, 
                                           self.params.max_velocity)
            
            # Step 7: Update positions
            self.positions += self.velocities
            
            # Step 8: Check boundaries
            self.positions = np.clip(self.positions, -1, 1)
            
            # Store best pressure for history
            if self.archive:
                archive_objectives = [self.evaluate_objectives(pos) for pos in self.archive]
                best_pressure = min(obj[0] for obj in archive_objectives)  # First objective (sphere function)
                self.pressure_history.append(best_pressure)
            
            # Print progress
            if (iteration + 1) % 100 == 0:
                current_best = min(self.objectives[:, 0])  # First objective
                print(f'Iteration {iteration + 1}/{self.params.max_iterations}')
                print(f'Current best pressure: {current_best:.10e}')
                print(f'Archive size: {len(self.archive)}')

    def plot_convergence(self):
        """Plot convergence history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.pressure_history)
        plt.yscale('log')
        plt.ylabel('Best Pressure (log scale)')
        plt.xlabel('Iteration')
        plt.title('Pareto-based WDO Convergence for Sphere Function')
        plt.grid(True)
        plt.show()

    def plot_pareto_front(self):
        """Plot final Pareto front"""
        if self.archive:
            objectives = np.array([self.evaluate_objectives(pos) for pos in self.archive])
            plt.figure(figsize=(10, 6))
            plt.scatter(objectives[:, 0], objectives[:, 1], c='b')
            plt.xlabel('Sphere Function Value')
            plt.ylabel('Solution Spread')
            plt.title('Final Pareto Front')
            plt.grid(True)
            plt.show()

def run_experiment(num_runs=1):
    """Run multiple experiments and show statistics"""
    best_pressures = []
    best_solutions = []
    
    for run in range(num_runs):
        print(f'\nRun {run + 1}/{num_runs}')
        
        np.random.seed(run)
        params = WDOParams()
        wdo = WindDrivenOptimization(params)
        
        wdo.optimize()
        
        if wdo.archive:
            # Get best solution based on first objective (sphere function)
            archive_objectives = [wdo.evaluate_objectives(pos) for pos in wdo.archive]
            best_idx = np.argmin([obj[0] for obj in archive_objectives])
            best_position = wdo.archive[best_idx]
            best_pressure = archive_objectives[best_idx][0]
            
            x_best = ((params.dim_max - params.dim_min) * (best_position + 1) / 2) + params.dim_min
            best_pressures.append(best_pressure)
            best_solutions.append(x_best)
        
        # Plot for last run
        if run == num_runs - 1:
            wdo.plot_convergence()
            wdo.plot_pareto_front()
    
    # Print statistics
    if best_pressures:
        print('\nExperiment Results:')
        print(f'Number of runs: {num_runs}')
        print(f'Average best pressure: {np.mean(best_pressures):.10e}')
        print(f'Best pressure found: {np.min(best_pressures):.10e}')
        print(f'Worst pressure found: {np.max(best_pressures):.10e}')
        print(f'Standard deviation: {np.std(best_pressures):.10e}')
        
        best_run = np.argmin(best_pressures)
        print(f'\nBest solution found (Run {best_run + 1}):')
        print(f'x* = {best_solutions[best_run]}')
        print(f'F(x*) = {best_pressures[best_run]:.10e}')

if __name__ == "__main__":
    run_experiment(num_runs=5)
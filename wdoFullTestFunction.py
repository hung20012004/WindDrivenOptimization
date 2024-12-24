import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TestFunctions:
    @staticmethod
    def sphere(x):
        """Hàm Sphere: F_SPH(x) = Σ(xi - 2)²"""
        return np.sum(np.power(x - 2, 2))

    @staticmethod
    def rotated_hyper_ellipsoid(x):
        """Hàm Rotated Hyper Ellipsoid: F_RHE(x) = Σ(Σxj)²"""
        n = len(x)
        sum_total = 0
        for i in range(n):
            sum_inner = sum(x[j] for j in range(i+1))
            sum_total += sum_inner**2
        return sum_total

    @staticmethod
    def ackley(x):
        """Hàm Ackley"""
        n = len(x)
        sum_sq = np.sum(np.power(x, 2))
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(sum_sq/n)) - np.exp(sum_cos/n)

    @staticmethod
    def rastrigin(x):
        """Hàm Rastrigin: F_RAS(x) = 10N + Σ(xi² - 10cos(2πxi))"""
        n = len(x)
        return 10 * n + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def six_hump_camel_back(x):
        """Hàm Six-Hump Camel-Back (chỉ cho 2 chiều)"""
        x1, x2 = x[0], x[1]
        return (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2

class WDOConfig:
    def __init__(self, function_name):
        self.function_name = function_name
        # Cấu hình chung
        self.population_size = 30
        self.max_iterations = 1000
        self.RT = 3

        # Cấu hình theo từng hàm từ bảng 2
        configs = {
            'sphere': {'alpha': 0.75, 'vmax': 0.25, 'gravity': 0.700, 'coriolis': -8},
            'rotated_hyper_ellipsoid': {'alpha': 0.75, 'vmax': 0.15, 'gravity': 0.001, 'coriolis': -2},
            'ackley': {'alpha': 0.75, 'vmax': 0.10, 'gravity': 0.080, 'coriolis': -4},
            'rastrigin': {'alpha': 0.75, 'vmax': 0.20, 'gravity': 0.100, 'coriolis': -8},
            'six_hump_camel_back': {'alpha': 0.75, 'vmax': 0.15, 'gravity': 0.01, 'coriolis': -2}
        }

        # Thiết lập khoảng giới hạn và số chiều theo từng hàm
        ranges = {
            'sphere': {'dim': 5, 'min': -10, 'max': 10},
            'rotated_hyper_ellipsoid': {'dim': 5, 'min': -100, 'max': 100},
            'ackley': {'dim': 5, 'min': -32, 'max': 32},
            'rastrigin': {'dim': 5, 'min': -5, 'max': 5},
            'six_hump_camel_back': {'dim': 2, 'min': -5, 'max': 5}
        }

        # Áp dụng cấu hình
        config = configs[function_name]
        range_config = ranges[function_name]
        
        self.alpha = config['alpha']
        self.max_velocity = config['vmax']
        self.gravity = config['gravity']
        self.coriolis = config['coriolis']
        self.dimensions = range_config['dim']
        self.dim_min = range_config['min']
        self.dim_max = range_config['max']

class WindDrivenOptimization:
    def __init__(self, config):
        self.config = config
        self.positions = 2 * (np.random.rand(config.population_size, config.dimensions) - 0.5)
        self.velocities = config.max_velocity * 2 * (np.random.rand(config.population_size, config.dimensions) - 0.5)
        self.pressures = np.zeros(config.population_size)
        self.best_pressure = float('inf')
        self.best_position = None
        self.pressure_history = np.zeros(config.max_iterations)
        
        # Chọn hàm đánh giá
        self.test_functions = TestFunctions()
        self.evaluate_function = getattr(TestFunctions, config.function_name)

    def evaluate_pressure(self, position):
        """Chuyển đổi vị trí chuẩn hóa sang không gian thực và tính giá trị hàm"""
        x = ((self.config.dim_max - self.config.dim_min) * (position + 1) / 2) + self.config.dim_min
        return self.evaluate_function(x)

    def optimize(self):
        """Thực hiện quá trình tối ưu hóa"""
        for i in range(self.config.population_size):
            self.pressures[i] = self.evaluate_pressure(self.positions[i])
        
        self.best_pressure = np.min(self.pressures)
        best_idx = np.argmin(self.pressures)
        self.best_position = self.positions[best_idx].copy()
        
        for iteration in range(self.config.max_iterations):
            for i in range(self.config.population_size):
                random_perm = np.random.permutation(self.config.dimensions)
                velocity_other = self.velocities[i, random_perm]
                
                self.velocities[i] = ((1 - self.config.alpha) * self.velocities[i] - 
                                    (self.config.gravity * self.positions[i]) + 
                                    abs((1 / (i + 1)) - 1) * ((self.best_position - self.positions[i]) * self.config.RT) + 
                                    (self.config.coriolis * velocity_other / (i + 1)))
                
                self.velocities[i] = np.clip(self.velocities[i], -self.config.max_velocity, self.config.max_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], -1, 1)
                
                new_pressure = self.evaluate_pressure(self.positions[i])
                self.pressures[i] = new_pressure
                
                if new_pressure < self.best_pressure:
                    self.best_pressure = new_pressure
                    self.best_position = self.positions[i].copy()
            
            self.pressure_history[iteration] = self.best_pressure
            
            sort_idx = np.argsort(self.pressures)
            self.positions = self.positions[sort_idx]
            self.velocities = self.velocities[sort_idx]

    def plot_convergence(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.pressure_history)
        plt.ylabel('Pressure')
        plt.xlabel('Iteration')
        
        # Chỉ dùng thang logarit cho các hàm có giá trị dương
        if np.all(self.pressure_history > 0):
            plt.yscale('log')
        
        plt.title(f'WDO Convergence History for {self.config.function_name.replace("_", " ").title()} Function')
        plt.grid(True)
        plt.show()
def run_test(function_name):
    print(f'\nTesting {function_name.replace("_", " ").title()} Function:')
    np.random.seed(42)
    
    config = WDOConfig(function_name)
    wdo = WindDrivenOptimization(config)
    wdo.optimize()
    
    x_best = ((config.dim_max - config.dim_min) * (wdo.best_position + 1) / 2) + config.dim_min
    
    print(f'Best Pressure: {wdo.best_pressure}')
    print(f'Best Position: {x_best}')
    wdo.plot_convergence()
    return wdo.best_pressure, x_best

if __name__ == "__main__":
    # Test từng hàm
    functions = ['sphere', 'rotated_hyper_ellipsoid', 'ackley', 'rastrigin', 'six_hump_camel_back']
    
    for func in functions:
        run_test(func)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class WDOParams:
    def __init__(self):
        # Các tham số từ bảng 2
        self.population_size = 200
        self.dimensions = 5
        self.max_iterations = 1000
        self.RT = 3
        self.gravity = 0.700        # g = 0.700 từ bảng
        self.alpha = 0.75          # α = 0.75 từ bảng
        self.coriolis = -8         # c = -8 từ bảng
        self.max_velocity = 0.25   # Vmax = 0.25 từ bảng
        
        # Khoảng giá trị từ bảng 1 cho hàm Sphere
        self.dim_min = -10         # xi ∈ [-10,10]
        self.dim_max = 10
        self.x_optimal = 2.0       # x* = (2,2,...,2)

class WindDrivenOptimization:
    def __init__(self, params):
        self.params = params
        
        # Khởi tạo vị trí và vận tốc ngẫu nhiên cho quần thể
        self.positions = 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        self.velocities = params.max_velocity * 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        
        self.pressures = np.zeros(params.population_size)
        self.best_pressure = float('inf')
        self.best_position = None
        self.pressure_history = np.zeros(params.max_iterations)
    
    def evaluate_pressure(self, position):
        """Hàm Sphere từ bảng 1"""
        x = ((self.params.dim_max - self.params.dim_min) * (position + 1) / 2) + self.params.dim_min
        return np.sum(np.power((x - self.params.x_optimal), 2))  # (xi - 2)^2

    def optimize(self):
        # Đánh giá quần thể ban đầu
        for i in range(self.params.population_size):
            self.pressures[i] = self.evaluate_pressure(self.positions[i])
        
        # Tìm vị trí tốt nhất ban đầu
        self.best_pressure = np.min(self.pressures)
        best_idx = np.argmin(self.pressures)
        self.best_position = self.positions[best_idx].copy()
        
        # Vòng lặp tối ưu
        for iteration in range(self.params.max_iterations):
            for i in range(self.params.population_size):
                # Hiệu ứng Coriolis
                random_perm = np.random.permutation(self.params.dimensions)
                velocity_other = self.velocities[i, random_perm]
                
                # Cập nhật vận tốc theo công thức WDO với tham số từ bảng
                self.velocities[i] = ((1 - self.params.alpha) * self.velocities[i] - 
                                    (self.params.gravity * self.positions[i]) + 
                                    abs((1 / (i + 1)) - 1) * ((self.best_position - self.positions[i]) * self.params.RT) + 
                                    (self.params.coriolis * velocity_other / (i + 1)))
                
                # Giới hạn vận tốc
                self.velocities[i] = np.clip(self.velocities[i], -self.params.max_velocity, self.params.max_velocity)
                
                # Cập nhật vị trí
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], -1, 1)
                
                # Đánh giá vị trí mới
                new_pressure = self.evaluate_pressure(self.positions[i])
                self.pressures[i] = new_pressure
                
                # Cập nhật vị trí tốt nhất nếu tìm thấy áp suất thấp hơn
                if new_pressure < self.best_pressure:
                    self.best_pressure = new_pressure
                    self.best_position = self.positions[i].copy()
            
            self.pressure_history[iteration] = self.best_pressure
            
            # Sắp xếp quần thể theo áp suất
            sort_idx = np.argsort(self.pressures)
            self.positions = self.positions[sort_idx]
            self.velocities = self.velocities[sort_idx]

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.pressure_history)
        plt.ylabel('Pressure (log scale)')
        plt.xlabel('Iteration')
        plt.yscale('log')
        plt.title('WDO Convergence History for Sphere Function')
        plt.grid(True)
        plt.show()

# Chạy thuật toán
if __name__ == "__main__":
    np.random.seed(42)  # Để kết quả có thể lặp lại
    
    print(f'Bắt đầu tối ưu hóa hàm Sphere...')
    print(f'Giá trị tối ưu lý thuyết: x* = (2,2,...,2), F(x*) = 0')
    
    params = WDOParams()
    wdo = WindDrivenOptimization(params)
    wdo.optimize()
    
    print(f'\nKết quả:')
    print(f'Áp suất tốt nhất: {wdo.best_pressure}')
    x_best = ((params.dim_max - params.dim_min) * (wdo.best_position + 1) / 2) + params.dim_min
    print(f'Vị trí tốt nhất: {x_best}')
    
    wdo.plot_convergence()
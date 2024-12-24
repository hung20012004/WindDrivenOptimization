import numpy as np
import matplotlib.pyplot as plt

class WingParams:
    def __init__(self):
        self.population_size = 200
        self.dimensions = 4  # góc tấn, độ cong, độ dày, hình dạng mép
        self.max_iterations = 1000
        self.RT = 3
        self.gravity = 0.7
        self.alpha = 0.75
        self.coriolis = -8
        self.max_velocity = 0.25
        
        # Giới hạn các thông số
        self.angle_range = (-5, 15)     # góc tấn từ -5 đến 15 độ
        self.camber_range = (0, 0.15)   # độ cong 0-15% dây cung
        self.thick_range = (0.08, 0.15) # độ dày 8-15% dây cung
        self.edge_range = (0, 1)        # hệ số hình dạng mép 0-1

class WingOptimization:
    def __init__(self, params):
        self.params = params
        self.positions = 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        self.velocities = params.max_velocity * 2 * (np.random.rand(params.population_size, params.dimensions) - 0.5)
        self.pressures = np.zeros(params.population_size)
        self.best_pressure = float('inf')
        self.best_position = None
        self.pressure_history = []

    def evaluate_pressure(self, position):
        """Đánh giá chất lượng thiết kế cánh"""
        # Chuyển đổi vị trí chuẩn hóa sang thông số thực
        angle = self._normalize(position[0], *self.params.angle_range)
        camber = self._normalize(position[1], *self.params.camber_range)
        thickness = self._normalize(position[2], *self.params.thick_range)
        edge_shape = self._normalize(position[3], *self.params.edge_range)
        
        # Giả lập tính toán CFD
        cl = self._calculate_lift(angle, camber)           # Hệ số nâng
        cd = self._calculate_drag(angle, thickness)        # Hệ số cản
        structural = self._structural_penalty(thickness)    # Phạt về cấu trúc
        
        # Hàm mục tiêu: Minimize drag với ràng buộc về lift
        pressure = cd
        if cl < 1.0:
            pressure += 10 * (1.0 - cl)**2 
        pressure += structural
        return pressure

    def _normalize(self, value, min_val, max_val):
        """Chuyển đổi từ [-1,1] sang range thực"""
        return ((max_val - min_val) * (value + 1) / 2) + min_val

    def _calculate_lift(self, angle, camber):
        return 0.2 * angle + 4.0 * camber + 0.3  # Tăng độ nhạy

    def _calculate_drag(self, angle, thickness):
        return 0.005 * angle**2 + 0.05 * thickness  

    def _structural_penalty(self, thickness):
        """Phạt nếu độ dày quá nhỏ"""
        if thickness < 0.1:
            return 10 * (0.1 - thickness)**2
        return 0

    def optimize(self):
        # Khởi tạo đánh giá
        for i in range(self.params.population_size):
            self.pressures[i] = self.evaluate_pressure(self.positions[i])
        self.best_pressure = np.min(self.pressures)
        best_idx = np.argmin(self.pressures)
        self.best_position = self.positions[best_idx].copy()

        # Vòng lặp tối ưu
        for iteration in range(self.params.max_iterations):
            for i in range(self.params.population_size):
                random_perm = np.random.permutation(self.params.dimensions)
                velocity_other = self.velocities[i, random_perm]
                
                # Cập nhật vận tốc
                self.velocities[i] = ((1 - self.params.alpha) * self.velocities[i] - 
                                    (self.params.gravity * self.positions[i]) + 
                                    abs((1 / (i + 1)) - 1) * ((self.best_position - self.positions[i]) * self.params.RT) + 
                                    (self.params.coriolis * velocity_other / (i + 1)))
                
                self.velocities[i] = np.clip(self.velocities[i], -self.params.max_velocity, self.params.max_velocity)
                
                # Cập nhật vị trí
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], -1, 1)
                
                # Đánh giá thiết kế mới
                new_pressure = self.evaluate_pressure(self.positions[i])
                self.pressures[i] = new_pressure
                
                if new_pressure < self.best_pressure:
                    self.best_pressure = new_pressure
                    self.best_position = self.positions[i].copy()
            
            self.pressure_history.append(self.best_pressure)

    def get_best_design(self):
        """Trả về thiết kế tối ưu"""
        return {
            'angle': self._normalize(self.best_position[0], *self.params.angle_range),
            'camber': self._normalize(self.best_position[1], *self.params.camber_range),
            'thickness': self._normalize(self.best_position[2], *self.params.thick_range),
            'edge_shape': self._normalize(self.best_position[3], *self.params.edge_range),
            'drag_coefficient': self.best_pressure
        }

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.pressure_history)
        plt.ylabel('Drag Coefficient')
        plt.xlabel('Iteration')
        plt.yscale('log')
        plt.title('Wing Design Optimization Convergence')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    params = WingParams()
    optimizer = WingOptimization(params)
    optimizer.optimize()
    
    best_design = optimizer.get_best_design()
    print("\nThiết kế cánh tối ưu:")
    print(f"Góc tấn: {best_design['angle']:.2f}°")
    print(f"Độ cong: {best_design['camber']:.3f}")
    print(f"Độ dày: {best_design['thickness']:.3f}")
    print(f"Hệ số hình dạng mép: {best_design['edge_shape']:.3f}")
    print(f"Hệ số cản: {best_design['drag_coefficient']:.6f}")
    
    optimizer.plot_convergence()
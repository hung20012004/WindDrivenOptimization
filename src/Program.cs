using System;
using System.Linq;

public class WindDrivenOptimization
{
    private readonly int dimensions;
    private readonly int populationSize;
    private readonly int maxIterations;
    private readonly double alpha;     // Friction coefficient
    private readonly double g;          // Gravitational acceleration
    private readonly double c;          // Coriolis constant
    private readonly double vMax;       // Maximum velocity
    private readonly Random random;

    private class AirParcel
    {
        public double[] Position { get; set; }
        public double[] Velocity { get; set; }
        public double Pressure { get; set; }
    }

    public WindDrivenOptimization(
        int dimensions, 
        int populationSize = 200,   
        int maxIterations = 5000,   
        double alpha = 0.75,         
        double g = 0.001,             
        double c = -2,              
        double vMax = 0.15)          
    {
        this.dimensions = dimensions;
        this.populationSize = populationSize;
        this.maxIterations = maxIterations;
        this.alpha = alpha;
        this.g = g;
        this.c = c;
        this.vMax = vMax;
        this.random = new Random();
    }

    public delegate double FitnessFunction(double[] position);

    public double[] Optimize(FitnessFunction fitnessFunction, double[] lowerBounds, double[] upperBounds)
    {
        var population = InitializePopulation(lowerBounds, upperBounds);

        // Compute initial pressure for population
        foreach (var parcel in population)
        {
            parcel.Pressure = -fitnessFunction(parcel.Position);
        }

        var bestParcel = population.OrderByDescending(p => p.Pressure).First();
        var globalBestPosition = bestParcel.Position.Clone() as double[];
        double globalBestPressure = bestParcel.Pressure;

        // Để lưu trữ avg pressure
        double[] avgPressureHistory = new double[maxIterations];

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            foreach (var parcel in population)
            {
                UpdateVelocity(parcel, bestParcel, globalBestPosition);
                UpdatePosition(parcel, lowerBounds, upperBounds);
                
                // Recompute pressure
                parcel.Pressure = -fitnessFunction(parcel.Position);
            }

            // Tính avg pressure
            double avgPressure = population.Average(p => p.Pressure);
            avgPressureHistory[iteration] = avgPressure;

            // Tìm parcel có pressure cao nhất
            bestParcel = population.OrderByDescending(p => p.Pressure).First();

            // Cập nhật global best nếu tìm được pressure tốt hơn
            if (bestParcel.Pressure > globalBestPressure)
            {
                globalBestPressure = bestParcel.Pressure;
                globalBestPosition = bestParcel.Position.Clone() as double[];
            }
        }

        // In ra avg pressure của các iteration cuối
        Console.WriteLine("\nAvg Pressure tại các iteration cuối:");
        for (int i = Math.Max(0, maxIterations - 10); i < maxIterations; i++)
        {
            Console.WriteLine($"Iteration {i}: {avgPressureHistory[i]:E6}");
        }

        Console.WriteLine($"\nGlobal Best Pressure: {globalBestPressure:E6}");

        return globalBestPosition;
    }

    private AirParcel[] InitializePopulation(double[] lowerBounds, double[] upperBounds)
    {
        var population = new AirParcel[populationSize];

        for (int i = 0; i < populationSize; i++)
        {
            population[i] = new AirParcel
            {
                Position = new double[dimensions],
                Velocity = new double[dimensions]
            };

            for (int j = 0; j < dimensions; j++)
            {
                population[i].Position[j] = lowerBounds[j] + 
                    random.NextDouble() * (upperBounds[j] - lowerBounds[j]);
                
                population[i].Velocity[j] = (random.NextDouble() * 2 - 1) * vMax;
            }
        }

        return population;
    }

    private void UpdateVelocity(AirParcel parcel, AirParcel bestParcel, double[] globalBestPosition)
    {
        for (int i = 0; i < dimensions; i++)
        {
            double friction = (1 - alpha) * parcel.Velocity[i];
            double gravitation = -g * parcel.Position[i];
            
            double pressureGradient = 
                0.5 * (bestParcel.Position[i] - parcel.Position[i]) / 
                (Math.Abs(bestParcel.Pressure) + 1e-10);
            
            double coriolisLikeTerm = c * (globalBestPosition[i] - parcel.Position[i]);

            parcel.Velocity[i] = friction + gravitation + pressureGradient + coriolisLikeTerm;

            parcel.Velocity[i] = Math.Clamp(parcel.Velocity[i], -vMax, vMax);
        }
    }

    private void UpdatePosition(AirParcel parcel, double[] lowerBounds, double[] upperBounds)
    {
        for (int i = 0; i < dimensions; i++)
        {
            parcel.Position[i] += parcel.Velocity[i];

            parcel.Position[i] = Math.Clamp(
                parcel.Position[i], 
                lowerBounds[i], 
                upperBounds[i]
            );
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        // DemoSphereFunctionOptimization();
        // DemoAckleyFunctionOptimization();
        // DemoRastriginFunctionOptimization();
        DemoRotatedHyperEllipsoidFunctionOptimization();
        // DemoSixHumpCamelBackFunctionOptimization();
    }

    static void DemoSphereFunctionOptimization()
    {
        var wdo = new WindDrivenOptimization(
            dimensions: 10, 
            populationSize: 200, 
            maxIterations: 5000
        );

        WindDrivenOptimization.FitnessFunction sphereFunction = (double[] x) => 
        {
            return x.Sum(val => val * val);  // Sphere function, global minimum at (0,0,...,0)
        };

        double[] lowerBounds = Enumerable.Repeat(-10.0, 10).ToArray();
        double[] upperBounds = Enumerable.Repeat(10.0, 10).ToArray();

        double[] bestSolution = wdo.Optimize(sphereFunction, lowerBounds, upperBounds);

        Console.WriteLine("\nGiải pháp tốt nhất cho hàm Sphere 10 chiều:");
        Console.WriteLine(string.Join(", ", bestSolution.Select(x => x.ToString("F4"))));
        Console.WriteLine($"Giá trị fitness: {sphereFunction(bestSolution):F4}");
    }
    static void DemoRotatedHyperEllipsoidFunctionOptimization()
    {
        var wdo = new WindDrivenOptimization(
            dimensions: 10,
            populationSize: 200,
            maxIterations: 5000
        );

        WindDrivenOptimization.FitnessFunction fitnessFunction = (double[] x) =>
        {
            return FRHE(x);
        };

        double[] lowerBounds = Enumerable.Repeat(-100.0, 10).ToArray();
        double[] upperBounds = Enumerable.Repeat(100.0, 10).ToArray();

        double[] bestSolution = wdo.Optimize(fitnessFunction, lowerBounds, upperBounds);

        Console.WriteLine("\nGiải pháp tốt nhất cho hàm Rotated Hyper Ellipsoid:");
        Console.WriteLine(string.Join(", ", bestSolution.Select(x => x.ToString("F4"))));
        Console.WriteLine($"Giá trị fitness: {fitnessFunction(bestSolution):F4}");
    }

    static void DemoAckleyFunctionOptimization()
    {
        var wdo = new WindDrivenOptimization(
            dimensions: 10,
            populationSize: 200,
            maxIterations: 5000
        );

        WindDrivenOptimization.FitnessFunction fitnessFunction = (double[] x) =>
        {
            return FACK(x);
        };

        double[] lowerBounds = Enumerable.Repeat(-32.0, 10).ToArray();
        double[] upperBounds = Enumerable.Repeat(32.0, 10).ToArray();

        double[] bestSolution = wdo.Optimize(fitnessFunction, lowerBounds, upperBounds);

        Console.WriteLine("\nGiải pháp tốt nhất cho hàm Ackley:");
        Console.WriteLine(string.Join(", ", bestSolution.Select(x => x.ToString("F4"))));
        Console.WriteLine($"Giá trị fitness: {fitnessFunction(bestSolution):F4}");
    }

    static void DemoRastriginFunctionOptimization()
    {
        var wdo = new WindDrivenOptimization(
            dimensions: 10,
            populationSize: 200,
            maxIterations: 5000
        );

        WindDrivenOptimization.FitnessFunction fitnessFunction = (double[] x) =>
        {
            return FRAS(x);
        };

        double[] lowerBounds = Enumerable.Repeat(-5.0, 10).ToArray();
        double[] upperBounds = Enumerable.Repeat(5.0, 10).ToArray();

        double[] bestSolution = wdo.Optimize(fitnessFunction, lowerBounds, upperBounds);

        Console.WriteLine("\nGiải pháp tốt nhất cho hàm Rastrigin:");
        Console.WriteLine(string.Join(", ", bestSolution.Select(x => x.ToString("F4"))));
        Console.WriteLine($"Giá trị fitness: {fitnessFunction(bestSolution):F4}");
    }

    static void DemoSixHumpCamelBackFunctionOptimization()
    {
        var wdo = new WindDrivenOptimization(
            dimensions: 2,
            populationSize: 200,
            maxIterations: 5000
        );

        WindDrivenOptimization.FitnessFunction fitnessFunction = (double[] x) =>
        {
            return FCB(x);
        };

        double[] lowerBounds = { -5.0, -5.0 };
        double[] upperBounds = { 5.0, 5.0 };

        double[] bestSolution = wdo.Optimize(fitnessFunction, lowerBounds, upperBounds);

        Console.WriteLine("\nGiải pháp tốt nhất cho hàm Six-Hump Camel-Back:");
        Console.WriteLine(string.Join(", ", bestSolution.Select(x => x.ToString("F4"))));
        Console.WriteLine($"Giá trị fitness: {fitnessFunction(bestSolution):F4}");
    }
    private static double FRHE(double[] x)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double innerSum = 0;
            for (int j = 0; j <= i; j++)
            {
                innerSum += Math.Pow(x[j], 2);
            }
            sum += Math.Pow(innerSum, 2);
        }
        return sum;
    }

    private static double FACK(double[] x)
    {
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum1 += Math.Pow(x[i], 2);
            sum2 += Math.Cos(2 * Math.PI * x[i]);
        }
        return 20 + Math.E - 20 * Math.Exp(-0.2 * Math.Sqrt(sum1 / x.Length)) - Math.Exp(sum2 / x.Length);
    }

    private static double FRAS(double[] x)
    {
        double sum = 10 * x.Length;
        for (int i = 0; i < x.Length; i++)
        {
            sum += Math.Pow(x[i], 2) - 10 * Math.Cos(2 * Math.PI * x[i]);
        }
        return sum;
    }

    private static double FCB(double[] x)
    {
        double x1 = x[0];
        double x2 = x[1];
        return 4 * Math.Pow(x1, 2) - 2.1 * Math.Pow(x1, 4) + Math.Pow(x1, 6) / 3 + x1 * x2 - 4 * Math.Pow(x2, 2) + 4 * Math.Pow(x2, 4);
    }
}
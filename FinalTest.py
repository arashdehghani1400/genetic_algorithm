import random

# Function to read the distance matrix from a file
def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

# Function to calculate the total distance of a given route
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    total_distance += distance_matrix[route[-1]][route[0]]  
    return total_distance

# greedy selection
def greedy_selection(population, fitness, num_selected):
    paired_population = list(zip(population, fitness))
    sorted_population = sorted(paired_population, key=lambda x: x[1], reverse=True)
    selected_individuals = [individual for individual, fit in sorted_population[:num_selected]]    
    return selected_individuals

# Function for Roulette Wheel Selection
def roulette_wheel_selection(population, fitnesses):
    max_fitness = sum(fitnesses)
    pick = random.uniform(0, max_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual

# Function for Single Point Crossover
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
    return child1, child2

# Function for Two Point Crossover
def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1_middle = parent2[point1:point2]
    child2_middle = parent1[point1:point2]
    
    child1 = parent1[:point1] + child1_middle + [gene for gene in parent1 if gene not in child1_middle]
    child2 = parent2[:point1] + child2_middle + [gene for gene in parent2 if gene not in child2_middle]
    
    return child1, child2

# Function for Single Point Mutation
def single_point_mutation(individual):
    idx1, idx2 = random.sample(range(1, len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Function for Multiple Point Mutation
def multiple_point_mutation(individual, num_mutations=2):
    for _ in range(num_mutations):
        idx1, idx2 = random.sample(range(1, len(individual)), 2)
        individual[idx1], idx2 = individual[idx2], individual[idx1]
    return individual

# Genetic Algorithm Framework
def genetic_algorithm(distance_matrix):
    cities = list(range(1, len(distance_matrix)))
    population_size = 100  
    generations = 500  
    mutation_rate = 0.1
    crossover_rate = 0.8 
    epsilon = 1e-6  

    # Initialize population
    population = [random.sample(cities, len(cities)) for _ in range(population_size)]

    for generation in range(generations):
        fitnesses = [1 / (calculate_total_distance([0] + individual, distance_matrix) + epsilon) for individual in population]

        new_population = []
        elite_size = 2
        elite_individuals = sorted(population, key=lambda ind: calculate_total_distance([0] + ind, distance_matrix))[:elite_size]
        new_population.extend(elite_individuals)

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)
            # parent1 = greedy_selection(population, fitnesses, 6)
            # parent2 = greedy_selection(population, fitnesses, 6)
            
            if random.random() < crossover_rate:
                # child1, child2 = single_point_crossover(parent1, parent2)
                child1, child2 = two_point_crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                child1 = single_point_mutation(child1)
                child2 = single_point_mutation(child2)

            # if random.random() < mutation_rate:
            #     child1 = multiple_point_mutation(child1)
            #     child2 = multiple_point_mutation(child2)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population[:population_size]

    best_route = min(population, key=lambda ind: calculate_total_distance([0] + ind, distance_matrix))
    best_distance = calculate_total_distance([0] + best_route, distance_matrix)

    return [0] + best_route , best_distance

# Main execution
distance_matrix_file = 'distance_matrix.txt'
distance_matrix = read_distance_matrix(distance_matrix_file)
best_route, best_distance = genetic_algorithm(distance_matrix)

route_str = ' -> '.join(chr(65 + city) for city in best_route) + f" -> {chr(65)}"
print(f"The shortest route found is: {route_str}: {best_distance}")

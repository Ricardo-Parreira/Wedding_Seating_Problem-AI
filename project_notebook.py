import itertools
import matplotlib.pyplot as plt
import math, copy
import random
from collections import defaultdict
from collections import Counter
import time
import numpy as np



table_arrangement = [[0,2], [1,3]]

test_preference_matrix = [
    [  0,  10,  -5,  20, -10,  15,   0,  -5],  # A
    [ 10,   0,  30,  -5,  20, -10,   5,   0],  # B
    [ -5,  30,   0,  10, -20,   5,  15,  10],  # C
    [ 20,  -5,  10,   0,  25,  -5, -15,  20],  # D
    [-10,  20, -20,  25,   0,  30, -10,   5],  # E
    [ 15, -10,   5,  -5,  30,   0,  20, -10],  # F
    [  0,   5,  15, -15, -10,  20,   0,  30],  # G
    [ -5,   0,  10,  20,   5, -10,  30,   0]   # H
] #ABCDABCD

def random_preferences(nguests):
    preference_matrix = [[0] * nguests for x in range(nguests)]
    for guest1 in range(nguests):
        for guest2 in range(nguests):
            if guest1 != guest2:
                score = random.randrange(-100, 100, 1)
                preference_matrix[guest1][guest2] = score
                preference_matrix[guest2][guest1] = score
    return preference_matrix


def evaluate_table(table, matrix):
    score = 0
    for guest in table:
        for neighbor in table:
            if guest != neighbor:
                score += matrix[guest][neighbor]

    return score

def standard_deviation(solution, preference_matrix):
    score_tables = []
    total_score=0
    for table in solution:
        current_score = evaluate_table(table, preference_matrix)
        total_score += current_score
        score_tables.append(current_score)
    average = total_score/len(solution)

    std_sum=0
    for score in score_tables:
        std_sum += (score-average)**2
    
    std = math.sqrt(std_sum/len(solution))
    return std

def evaluate_solution(solution, preference_matrix):
    average = average_tables(solution, preference_matrix)  #not optimal
    std = standard_deviation(solution, preference_matrix) + 1  # shift de 1 para nunca poder haver divisao por 0
    return (average/std)
    

def average_tables(tables, matrix):
    score = 0
    for table in tables:
        score += evaluate_table(table, matrix)
    average = score / len(tables)
    return average


def fill_matrix(seatsPerTable, matrix):
    nguests = len(matrix)
    total_tables = math.ceil(nguests / seatsPerTable) #rounds the division up to the next integer
    TotalSeats = total_tables * seatsPerTable
    diff = TotalSeats % nguests 

    matrix_copy = copy.deepcopy(matrix)
    if diff == 0:
        return matrix_copy
    else:
        #fill the matrix with the preference that the guests have of the empty seats
        for guest in range(len(matrix_copy)):
            matrix_copy[guest] = matrix_copy[guest] + ([0]*diff)

        #fill the matrix with the preference of the emptyseats
        for i in range(diff):
            matrix_copy.append([0]*(nguests+diff))

    return matrix_copy
def random_arrangement(matrix, seatsPerTable):
    matrix_copy = fill_matrix(seatsPerTable, matrix)

    nguests = len(matrix_copy)
    guest_list = list(range(nguests))
    arrangement = []
    for table in range(nguests//seatsPerTable):
        table = random.sample(guest_list, seatsPerTable)
        for seatedGuest in table:
            guest_list.remove(seatedGuest)

        arrangement.append(table)
    return arrangement

def generate_population(population_size, preference_matrix, seatsPerTable):
    solutions = []
    for i in range(population_size):
        solutions.append(random_arrangement(preference_matrix, seatsPerTable))
    return solutions


def get_neighbour(curr_arrangement):
    neighbor_arrangement = copy.deepcopy(curr_arrangement)
    table1 = np.random.choice(len(neighbor_arrangement))
    table2 = np.random.choice(len(neighbor_arrangement))
    while(table1 == table2):
        table2 = np.random.choice(len(neighbor_arrangement))
    person1 = np.random.choice(len(neighbor_arrangement[table1]))
    person2 = np.random.choice(len(neighbor_arrangement[table2]))
    neighbor_arrangement[table1][person1], neighbor_arrangement[table2][person2] = neighbor_arrangement[table2][person2], neighbor_arrangement[table1][person1]
    return neighbor_arrangement
#improvement not filled table play

def advanced_get_neighbour(curr_arrangement):
    percentage = 0.05   #adjustable
    totalSeats = len(curr_arrangement)
    #randomly chooses from 1 guest to 5% of total guest population to be swaped
    toBeChanged = random.choices(range(1, int(totalSeats*percentage)+2))
    
    neighbour_arrangement = curr_arrangement
    for nguests in range(toBeChanged[0]):
        neighbour_arrangement = get_neighbour(neighbour_arrangement)
    return neighbour_arrangement
import random
import copy

def perturb_solution(arrangement):
    # faz uma cópia da disposição atual
    new_arrangement = copy.deepcopy(arrangement)
    #flattening
    guests = [guest for table in new_arrangement for guest in table]
    
    # decide quantos convidados vão ser trocados (entre 10% e 30%)
    n_guests = len(guests)
    n_to_shuffle = random.randint(int(0.1 * n_guests), int(0.3 * n_guests))
    
    # seleciona aleatoriamente os convidados para embaralhar
    guests_to_shuffle = random.sample(guests, n_to_shuffle)
    
    # remove os convidados escolhidos das mesas
    for table in new_arrangement:
        table[:] = [guest for guest in table if guest not in guests_to_shuffle]
    
    # baralha a lista de convidados selecionados
    random.shuffle(guests_to_shuffle)
    
    # reinsere os convidados de forma aleatória nas mesas
    for guest in guests_to_shuffle:
        random.choice(new_arrangement).append(guest)
    
    return new_arrangement

def random_crossover(parent1, parent2):
    num_tables = len(parent1)

    #print(len(parent1))
    #print(len(parent2))
    
    mask = [random.choice([0, 1]) for _ in range(num_tables)]
    
    mask[random.randint(0, num_tables - 1)] = 0  
    mask[random.randint(0, num_tables - 1)] = 1 
    
    child1, child2 = [[] for _ in range(num_tables)], [[] for _ in range(num_tables)]
    assigned1, assigned2 = set(), set()

    random.shuffle(parent1)
    random.shuffle(parent2)
    
    for i in range(num_tables):
        #print(i)
        if mask[i] == 0:
            child1[i] = [guest for guest in parent1[i] if guest not in assigned1]
            if child1[i] is not None: 
                assigned1.update(child1[i])
            child2[i] = [guest for guest in parent2[i] if guest not in assigned2]
            if child2[i] is not None: 
                assigned2.update(child2[i])
        else:
            child1[i] = [guest for guest in parent2[i] if guest not in assigned1]
            if child1[i] is not None: 
                assigned1.update(child1[i])
            child2[i] = [guest for guest in parent1[i] if guest not in assigned2]
            if child2[i] is not None: 
                assigned2.update(child2[i])
    
    all_guests = set(sum(parent1, []) + sum(parent2, []))  
    remaining1 = list(all_guests - assigned1)
    remaining2 = list(all_guests - assigned2)
    
    def fill_tables(child, remaining, parent_ref, assigned_set):
        random.shuffle(remaining)
        
        for i in range(num_tables):
            missing_count = len(parent_ref[i]) - len(child[i])
            if missing_count > 0:
                for guest in remaining[:missing_count]:
                    if guest not in assigned_set:
                        child[i].append(guest)
                        assigned_set.add(guest)
                remaining = remaining[missing_count:]  

    fill_tables(child1, remaining1, parent1, assigned1)
    fill_tables(child2, remaining2, parent2, assigned2)
    
    return child1, child2


def midpoint_crossover(parent1, parent2):
    num_tables = len(parent1)
    cut = num_tables // 2  # Ponto de corte no meio
    
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    
    assigned1, assigned2 = set(sum(child1, [])), set(sum(child2, []))
    all_guests = set(sum(parent1, []) + sum(parent2, []))
    remaining1, remaining2 = list(all_guests - assigned1), list(all_guests - assigned2)
    
    def fill_tables(child, remaining, parent_ref, assigned_set):
        random.shuffle(remaining)
        for i in range(num_tables):
            missing_count = len(parent_ref[i]) - len(child[i])
            for _ in range(missing_count):
                if remaining:
                    guest = remaining.pop()
                    child[i].append(guest)
                    assigned_set.add(guest)
    
    fill_tables(child1, remaining1, parent1, assigned1)
    fill_tables(child2, remaining2, parent2, assigned2)
    
    return child1, child2

def simmulated_annealing(preferences, seatsPerTable):
    start_time = time.time()
    
    #primeiro arranjamos um estado inicial random e avaliamos
    iterations = 1000
    cooling = 0.99
    initial_state = random_arrangement(preferences, seatsPerTable)
    filled_preferences = fill_matrix(seatsPerTable, preferences)
    initial_score = evaluate_solution(initial_state, filled_preferences)

    temperature = standard_deviation(initial_state, filled_preferences)   #basicamente a nossa tolerância no que toca a aceitar soluçoes piores
    cooling = 0.99    #o quao rápido vai descendo essa tolerancia

    # Add tracking for best scores
    best_scores = [initial_score]
    current_scores = [initial_score]
    
    iterations_count = 0
    while iterations > 0:
        iterations_count += 1
        #depois arranjamos uma soluçao vizinha à inicial e avaliamos essa
        neighbour_state = advanced_get_neighbour(initial_state)
        neighbour_score = evaluate_solution(neighbour_state, filled_preferences)
        
        score_diff = initial_score - neighbour_score

        #se a soluçao for melhor, aceitamos
        if score_diff < 0:
            initial_state = neighbour_state
            initial_score = neighbour_score
        #se for pior aceitamos com uma certa probabilidade que depende da temperatura
        else:
            probability = math.exp(-score_diff / temperature)
            if random.random() < probability:
                initial_state = neighbour_state
                initial_score = neighbour_score
                
        temperature *= cooling
        iterations -= 1
        
        current_scores.append(initial_score)
        best_scores.append(max(best_scores[-1], initial_score))
        
        if iterations_count % 1000 == 0:
            print(f"Iteration {iterations_count}, Current score: {initial_score}, Best score: {best_scores[-1]}")
    
    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
    
    show_graph(best_scores, current_scores)
    
    return initial_state, initial_score
def tournament_select(population, preference_matrix, tournament_size, exclude=None):
    filtered_population = [ind for ind in population if ind != exclude]
    selected = random.sample(filtered_population, tournament_size)
    best_solution = max(selected, key=lambda s: evaluate_solution(s, preference_matrix))
    return best_solution

def roulette_select(population, preference_matrix, exclude=None):
    filtered_population = [ind for ind in population if ind != exclude]
    fitness_values = np.array([evaluate_solution(s, preference_matrix) for s in filtered_population])
    total_fitness = np.sum(fitness_values)
    rand_value = np.random.uniform(0, total_fitness)

    cumulative_sum = 0
    for i, fitness in enumerate(fitness_values):
        cumulative_sum += fitness
        if rand_value <= cumulative_sum:
            return filtered_population[i]
    


def mutation(parent, mutation_prob=0.2):
    if random.random() < mutation_prob:  
        return get_neighbour(parent)
    return parent


def genetic_algorithm_1(num_iterations, population_size, preference_matrix, seatsPerTable):
    start_time = time.time()
        
    filled_preference_matrix = fill_matrix(seatsPerTable, preference_matrix)
    population = generate_population(population_size, filled_preference_matrix, seatsPerTable)
    best_solution = population[0]
    best_score = evaluate_solution(population[0], filled_preference_matrix)
    
    best_scores = []
    all_scores = []
    avg_scores = []

    for solution in population:
        all_scores.append(evaluate_solution(solution, filled_preference_matrix))


    print(f"Initial solution: {best_solution}, score: {best_score}")

    while(num_iterations > 0):


        parent1 = roulette_select( population, filled_preference_matrix)
        parent2 = roulette_select( population, filled_preference_matrix, exclude=parent1)

        if parent2 is None:
            parent2 = parent1

    
        #estatisticas para grafico
        avg_score = np.mean(all_scores)
        avg_scores.append(avg_score)
        best_solution = max(population, key=lambda x: evaluate_solution(x, filled_preference_matrix))
        best_scores.append(evaluate_solution(best_solution, filled_preference_matrix))


        # Next generation Crossover and Mutation
        child1, child2 = midpoint_crossover(parent1, parent2)
        child1, child2 = mutation(child1), mutation(child2)
        
        population.append(child1)
        population.append(child2)
        all_scores.append(evaluate_solution(child1, filled_preference_matrix))
        all_scores.append(evaluate_solution(child2, filled_preference_matrix))
        population.sort(key=lambda sol: evaluate_solution(sol, filled_preference_matrix), reverse=True)
        population = population[:population_size]

        num_iterations-=1
    
    best_solution= population[0]
    best_score=evaluate_solution(best_solution, filled_preference_matrix)

    num_guests = len(preference_matrix)
    best_solution = [[guest for guest in table if guest < num_guests] for table in best_solution]

    print(f"  Final solution: {best_solution}, score: {best_score}")

    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.6f} segundos")

    show_graph(best_scores, avg_scores)
    return best_solution
def tabu_search(preferences, seats_per_table, max_iterations=300000, tabu_tenure=12, max_no_improve=700):
    import time, copy
    start_time = time.time()

    # preenche a matriz de preferências para lidar com assentos vazios
    padded_preferences = fill_matrix(seats_per_table, preferences)
    # gera uma disposição inicial aleatória
    current_arrangement = random_arrangement(preferences, seats_per_table)
    best_arrangement = copy.deepcopy(current_arrangement)

    # avalia a solução inicial
    current_score = evaluate_solution(current_arrangement, padded_preferences)
    best_score = current_score

    # inicializa listas tabu e de frequência
    tabu_list = {}
    frequency_list = {}
    iterations_no_improve = 0
    total_iterations = 0

    # listas para rastrear os scores
    best_scores = [best_score]
    current_scores = [current_score]

    while total_iterations < max_iterations and iterations_no_improve < max_no_improve:
        total_iterations += 1

        # gera vários vizinhos
        neighbors = [get_neighbour(current_arrangement) for _ in range(10)]
        neighbor_scores = [evaluate_solution(n, padded_preferences) for n in neighbors]

        # seleciona o melhor vizinho
        best_neighbor_idx = max(range(len(neighbors)), key=lambda i: neighbor_scores[i])
        neighbor_arrangement = neighbors[best_neighbor_idx]
        neighbor_score = neighbor_scores[best_neighbor_idx]

        # verifica se o vizinho está na lista tabu
        is_tabu = tuple(map(tuple, neighbor_arrangement)) in tabu_list

        # critério de aspiração: aceita se for tabu mas melhora o score
        if is_tabu and neighbor_score <= best_score:
            frequency_list[tuple(map(tuple, neighbor_arrangement))] = frequency_list.get(tuple(map(tuple, neighbor_arrangement)), 0) + 1

            # força diversificação se preso num ciclo
            if frequency_list[tuple(map(tuple, neighbor_arrangement))] > 5:
                for _ in range(5):
                    current_arrangement = perturb_solution(current_arrangement, percent=0.3)
                    current_score = evaluate_solution(current_arrangement, padded_preferences)
                frequency_list.clear()
                iterations_no_improve += 1
                current_scores.append(current_score)
                best_scores.append(best_score)
                continue

        # move para o vizinho
        current_arrangement = neighbor_arrangement
        current_score = neighbor_score

        # atualiza a lista tabu
        keys_to_remove = []
        for arrangement in list(tabu_list):
            tabu_list[arrangement] -= 1
            if tabu_list[arrangement] <= 0:
                keys_to_remove.append(arrangement)
        for key in keys_to_remove:
            del tabu_list[key]

        tabu_list[tuple(map(tuple, current_arrangement))] = tabu_tenure
        current_scores.append(current_score)

        # atualiza a melhor solução
        if current_score > best_score:
            best_arrangement = copy.deepcopy(current_arrangement)
            best_score = current_score
            iterations_no_improve = 0
            frequency_list.clear()
        else:
            iterations_no_improve += 1

        best_scores.append(best_score)

        # log a cada 100 iterações
        if total_iterations % 100 == 0:
            print(f"[{total_iterations}] score atual: {current_score:.3f} | melhor: {best_score:.3f} | sem melhorar: {iterations_no_improve}")

        # aplica perturbação a cada 1000 iterações
        if total_iterations % 1000 == 0:
            current_arrangement = perturb_solution(current_arrangement, percent=0.35)
            current_score = evaluate_solution(current_arrangement, padded_preferences)
            print(f"perturbação aplicada na iteração {total_iterations}")
    
    # remove convidados fictícios
    original_guests = len(preferences)
    final_arrangement = []
    for table in best_arrangement:
        real_guests = [guest for guest in table if guest < original_guests]
        if real_guests:
            final_arrangement.append(real_guests)

    avg_no_improve = iterations_no_improve / total_iterations if total_iterations > 0 else 0
    end_time = time.time()

    print(f"score final: {best_score:.3f}")
    print(f"tempo de execução: {end_time - start_time:.3f} segundos")
    show_graph(best_scores, current_scores)

    return final_arrangement, best_score, avg_no_improve


def solution_to_tables(solution):
    mesas = defaultdict(list)
    for convidado, mesa in enumerate(solution):
        mesas[mesa].append(convidado)
    return list(mesas.values())

def evaluate_table_(table, matrix):
    score = 0
    for guest in table:
        for neighbor in table:
            if guest != neighbor:
                score += matrix[guest][neighbor]

    return score

def standard_deviation_(tables, preference_matrix):
    solution = solution_to_tables(tables)
    score_tables = []
    total_score=0
    for table in solution:
        current_score = evaluate_table_(table, preference_matrix)
        total_score += current_score
        score_tables.append(current_score)
    average = total_score/len(solution)

    std_sum=0
    for score in score_tables:
        std_sum += (score-average)**2
    
    std = math.sqrt(std_sum/len(solution))
    return std

def evaluate_solution_(solution, preference_matrix):
    average = average_tables_(solution, preference_matrix)  #not optimal
    std = standard_deviation_(solution, preference_matrix) + 1  # shift de 1 para nunca poder haver divisao por 0
    return (average/std)
    

def average_tables_(solution, matrix):
    tables = solution_to_tables(solution)
    score = 0
    for table in tables:
        score += evaluate_table_(table, matrix)
    average = score / len(tables)
    return average


def generate_population_(pop_size, preference_matrix, seatsPerTable):
    num_guests = len(preference_matrix)
    num_tables = (num_guests + seatsPerTable - 1) // seatsPerTable
    population = []
    
    for _ in range(pop_size):
        guests = list(range(num_guests))
        random.shuffle(guests)
        individual = [0] * num_guests
        for i, guest in enumerate(guests):
            mesa = i // seatsPerTable
            individual[guest] = mesa
        population.append(individual)
    
    return population


def get_neighbour_(curr_arrangement):
    neighbor = curr_arrangement[:]
    num_guests = len(neighbor)

    # Escolher dois convidados diferentes
    guest1 = random.randint(0, num_guests - 1)
    guest2 = random.randint(0, num_guests - 1)
    while guest1 == guest2 or neighbor[guest1] == neighbor[guest2]:
        guest2 = random.randint(0, num_guests - 1)

    # Trocar as mesas atribuídas entre os dois convidados
    neighbor[guest1], neighbor[guest2] = neighbor[guest2], neighbor[guest1]

    return neighbor

def advanced_get_neighbour_(curr_arrangement):
    percentage = 0.05   #adjustable
    totalSeats = len(curr_arrangement)
    #randomly chooses from 1 guest to 5% of total guest population to be swaped
    toBeChanged = random.choices(range(1, int(totalSeats*percentage)+2))
    
    neighbour_arrangement = curr_arrangement
    for nguests in range(toBeChanged[0]):
        neighbour_arrangement = get_neighbour_(neighbour_arrangement)
    return neighbour_arrangement



def random_crossover_(parent1, parent2, preference_matrix, seatsPerTable):
    num_guests = len(parent1)
    num_tables = (num_guests + seatsPerTable - 1) // seatsPerTable

    cut = random.randint(1, num_guests - 2)

    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]

    child1 = optimize_child(child1, num_tables, seatsPerTable, preference_matrix)
    child2 = optimize_child(child2, num_tables, seatsPerTable, preference_matrix)

    return child1, child2

def midpoint_crossover_(parent1, parent2, preference_matrix, seatsPerTable):
    num_guests = len(parent1)
    num_tables = (num_guests + seatsPerTable - 1) // seatsPerTable

    cut = num_guests // 2  # Ponto de corte no meio

    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]

    child1 = optimize_child(child1, num_tables, seatsPerTable, preference_matrix)
    child2 = optimize_child(child2, num_tables, seatsPerTable, preference_matrix)

    return child1, child2


def optimize_child(child, num_tables, seatsPerTable, preference_matrix):
    table_counts = Counter(child)

    # Mesas com mais do que o permitido
    overfilled = {mesa: count for mesa, count in table_counts.items() if count > seatsPerTable}
    # Mesas com espaço livre
    underfilled = {mesa: seatsPerTable - table_counts.get(mesa, 0) for mesa in range(num_tables) if table_counts.get(mesa, 0) < seatsPerTable}

    if not overfilled:
        return child  # solução já está válida

    # Identificar os convidados a mover (os que menos contribuem)
    guest_to_move = []
    for mesa in overfilled:
        guests = [i for i, m in enumerate(child) if m == mesa]
        guests_sorted = sorted(guests, key=lambda g: contribution_to_table(g, child, preference_matrix), reverse=True)
        needed = overfilled[mesa] - seatsPerTable
        guest_to_move.extend(guests_sorted[-needed:])  # mover os que contribuem menos

    # Mover os convidados para mesas com espaço
    underfilled_list = list(underfilled.items())  # [(mesa, lugares)]
    idx = 0
    for guest in guest_to_move:
        while idx < len(underfilled_list) and underfilled_list[idx][1] == 0:
            idx += 1
        if idx >= len(underfilled_list):
            break  # tudo alocado
        mesa_destino = underfilled_list[idx][0]
        child[guest] = mesa_destino
        underfilled_list[idx] = (mesa_destino, underfilled_list[idx][1] - 1)

    return child


def contribution_to_table(guest, solution, matrix):
    mesa = solution[guest]
    same_table = [i for i in range(len(solution)) if i != guest and solution[i] == mesa]
    return sum(matrix[guest][other] + matrix[other][guest] for other in same_table)


def tournament_select_(population, preference_matrix, tournament_size, exclude=None):
    filtered_population = [ind for ind in population if ind != exclude]

    if len(filtered_population) == 0:
        return exclude 

    #print("pop size: ")
    #print(len(filtered_population))
    # Corrigir o tamanho do torneio para nunca ultrapassar o tamanho da população
    tournament_size = min(tournament_size, len(filtered_population))

    selected = random.sample(filtered_population, tournament_size)
    best_solution = max(selected, key=lambda s: evaluate_solution_(s, preference_matrix))
    return best_solution


def roulette_select_(population, preference_matrix, exclude=None):
    filtered_population = [ind for ind in population if ind != exclude]
    fitness_values = np.array([evaluate_solution_(s, preference_matrix) for s in filtered_population])
    total_fitness = np.sum(fitness_values)
    rand_value = np.random.uniform(0, total_fitness)

    cumulative_sum = 0
    for i, fitness in enumerate(fitness_values):
        cumulative_sum += fitness
        if rand_value <= cumulative_sum:
            return filtered_population[i]
    

def mutation_(parent, mutation_prob=0.1):
    if random.random() < mutation_prob:  
        return get_neighbour_(parent)
    return parent


def genetic_algorithm_2(num_iterations, population_size, preference_matrix, seatsPerTable):
    start_time = time.time()
        
    
    filled_preference_matrix = fill_matrix(seatsPerTable, preference_matrix)
    population = generate_population_(population_size, filled_preference_matrix, seatsPerTable)
    """print("population[0]: ")
    print(population[0])"""
    best_solution = population[0]
    best_score = evaluate_solution_(population[0], filled_preference_matrix)
    #num_iterations=500

    best_scores = []
    all_scores = []
    avg_scores = []

    for solution in population:
        all_scores.append(evaluate_solution_(solution, filled_preference_matrix))
    

    print(f"Initial solution: {best_solution}, score: {best_score}")

    while(num_iterations > 0):

        #parent1 = tournament_select_( population, filled_preference_matrix, 10)
        parent1 = roulette_select_(population, filled_preference_matrix)
        #parent2 = tournament_select_( population, filled_preference_matrix, 10, parent1)
        parent2 = roulette_select_(population, filled_preference_matrix, exclude=parent1)

        if parent2 is None:
            parent2 = parent1

        #estatisticas para grafico
        avg_score = np.mean(all_scores)
        avg_scores.append(avg_score)
        best_solution = max(population, key=lambda x: evaluate_solution_(x, filled_preference_matrix))
        best_scores.append(evaluate_solution_(best_solution, filled_preference_matrix))


        # Next generation Crossover and Mutation
        child1, child2 = random_crossover_(parent1, parent2, filled_preference_matrix, seatsPerTable)


        child1, child2 = mutation_(child1), mutation_(child2)
        
        population.append(child1)
        population.append(child2)
        all_scores.append(evaluate_solution_(child1, filled_preference_matrix))
        all_scores.append(evaluate_solution_(child2, filled_preference_matrix))
        population.sort(key=lambda sol: evaluate_solution_(sol, filled_preference_matrix), reverse=True)
        population = population[:population_size]

        num_iterations-=1
    
    best_solution= population[0]
    best_score=evaluate_solution_(best_solution, filled_preference_matrix)
    print(f"  Final solution: {best_solution}, score: {best_score}")

    final_solution = solution_to_tables(best_solution)
    num_guests = len(preference_matrix) 
    best_solution = [[guest for guest in table if guest < num_guests] for table in final_solution]


    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
    show_graph(best_scores, avg_scores)

    return best_solution




def show_graph(best_scores, avg_scores):
    plt.plot(range(1, len(best_scores) + 1), best_scores, label='Best Individual Score')
    plt.plot(range(1, len(avg_scores) + 1), avg_scores, linestyle='--', color='red', label='Average Population Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Algorithm Performance')
    plt.legend()
    plt.show()



def run_wedding_seating(num_guests, seats_per_table, algorithm, matrix):

    matrix_ = [[0, -74, 56, -86, 60, 71, 22, 99, 89, 62, -58, 9, -66, 91, 9, 86, -47, -28, -74, 13, -80, 34, 56, -57, 54, 46, 54, 56, -36, 1, -3, -97, -46, 47, 71, -68, 52, -80, -97, 14, -48, -81, 71, -25, 33, 36, 62, 98, 91, -68, 30, -22, 30, 31, -2, -32, 15, 3, -32, 66, -9, 13, -29, -43, 39, -98, -46, -67, -44, -63, 82, -4, -100, -88, 39, 25, -76, -33, -78, 56, 45, 63, 42, -94, -29, 60, -96, 63, 15, 22, -65, -62, 18, -58, 66, -7, 64, -67, 30, -100], [-74, 0, 99, 11, -38, 70, -87, -76, -34, 43, -24, 36, -11, 55, -25, -70, 55, 70, -93, 94, 57, -39, -29, -76, 85, 88, 17, -98, 64, -65, 31, -67, -8, -94, -2, 42, -73, 66, -96, -99, -6, -81, -67, -56, 5, -50, 49, -88, -46, -21, 57, 83, -8, -23, 26, -31, -52, -33, -8, 33, 61, -41, -88, -94, 77, -64, 64, 3, 37, -62, -68, -86, 65, 61, -16, 11, 57, -78, 32, -95, 83, 13, 31, 4, 24, 63, -73, 30, -92, -98, 99, -8, -34, -75, 24, -12, 22, 48, -55, 88], [56, 99, 0, -98, 63, 42, 26, -93, -42, -38, -9, 86, -93, -59, 75, -22, -35, -55, -76, -13, -42, 67, 72, -13, 8, 8, -12, -21, -81, 92, 32, -71, 60, 72, -52, 90, -69, 33, -18, -76, 19, -59, 62, 14, -87, -65, 5, 68, -94, 79, -68, -59, -48, -51, 17, -60, -56, 72, 55, -18, 42, 57, 71, 9, -95, -62, 58, 20, 47, 61, -25, -79, 64, -97, 96, 53, 10, 6, -40, -4, 80, 26, -96, -14, -35, 35, -48, 73, -36, -4, -3, 50, -9, 17, -24, 68, 65, -46, 72, 29], [-86, 11, -98, 0, 19, 54, 88, 13, -13, 87, -78, 27, 83, 6, 18, 39, -73, -39, 5, -44, -88, -78, 12, -74, 44, -84, 21, -53, -9, -59, 68, 38, 81, 58, 63, -79, 49, 90, -80, 28, -32, -41, -70, -68, -90, -69, -64, -29, 64, -95, -23, 14, 39, -30, -20, 23, 80, 33, 69, 59, -44, 90, -62, -95, 55, -52, 29, 82, -7, -49, 63, -67, -52, 47, 7, -53, -36, 8, 98, 29, -31, 97, -23, -47, 65, 76, 85, 47, -32, 91, -85, -34, -62, -36, -1, -32, -74, -71, 63, 37], [60, -38, 63, 19, 0, 83, 51, 24, 51, -64, -83, 58, 93, 88, 97, -72, -4, 19, -32, 4, -80, -51, -59, -68, -28, 9, 17, 79, 54, 78, 85, 66, 32, -34, 97, -93, -69, 10, -40, 57, -94, 39, -36, 56, -74, -63, 3, 83, 72, 94, 13, -40, -79, 6, -81, -48, -74, 57, -45, -98, 83, -37, -82, -9, 80, -37, -7, 24, 7, -67, 11, 59, 46, -57, -55, -56, -35, -23, -7, -88, 89, -66, -97, -96, 53, -79, -11, 82, -78, -74, 45, 98, 95, 4, 53, 21, -75, 0, -78, 32], [71, 70, 42, 54, 83, 0, -84, -19, -38, -25, 39, 36, -17, 37, 41, 83, 30, 91, 8, 61, 51, -51, -31, 73, 2, -82, -82, -77, -23, 55, -31, -39, -89, -27, -71, -34, 2, -90, 41, -84, -59, 10, -31, -99, -82, -78, 85, -10, 92, 28, -87, -70, -14, -63, -41, -17, 71, 26, 26, -4, -4, -97, 24, 11, 38, -89, 1, 61, -28, -67, -34, 51, -7, -50, 88, -58, 68, 54, -79, -4, 92, 56, 30, -12, 17, 20, 93, 99, -47, -91, -70, -40, 48, -69, -3, -71, 36, -52, 37, -78], [22, -87, 26, 88, 51, -84, 0, 20, 50, -57, -18, 89, 65, 6, -8, -81, -34, 0, -71, -2, 87, 9, 8, -34, 59, -97, -85, 39, 63, 55, -73, 94, -39, -21, -80, -73, -63, -69, -84, 94, -2, -1, 52, -70, -60, -14, -72, -37, 92, -46, -56, -18, -75, -29, 59, 92, -77, -96, 28, 76, -66, 54, -88, 47, -21, -96, -68, 24, 27, -32, -61, -91, 86, -20, 92, 20, -38, -59, 14, -92, 19, 8, 76, -13, -81, -16, -86, 94, 21, 35, 12, -13, 61, -86, 19, 36, -63, -74, -32, -2], [99, -76, -93, 13, 24, -19, 20, 0, 8, 66, 9, -98, -87, 51, -49, -56, 70, -21, 99, 29, -56, 12, -10, 61, 82, -11, -30, 28, 4, 56, -18, 83, -51, -2, 36, 76, -56, -16, 43, 64, -36, -27, 55, 88, 8, -2, -65, 47, 64, 45, 64, -17, 29, -100, -50, 63, -43, -8, -1, 26, -75, 96, -82, -58, 35, 30, -100, -91, -37, 32, -94, -76, -69, -48, -21, 75, -32, -24, 62, 15, 86, -37, -74, -85, -25, 20, -77, -61, -98, -100, 14, 71, 23, 94, 35, -88, -26, 71, -79, -12], [89, -34, -42, -13, 51, -38, 50, 8, 0, -2, -33, 72, 2, -32, 23, 22, -43, 59, 41, 31, -91, 59, 3, -58, 36, 56, -43, 54, -16, 96, 23, 40, 30, 75, 0, 56, -19, 71, 94, -59, 73, -21, -39, -100, -35, 67, -57, -13, -30, -74, 7, 91, 70, 15, 89, 22, -53, 28, -29, 44, -41, 48, -18, 54, -13, 30, -88, 29, -68, 7, 92, 38, -41, -54, 56, -30, -14, -81, 28, 39, -33, -3, 58, 70, 50, 55, -3, 35, 45, 69, 56, -36, -22, 54, -8, -46, 82, 50, 34, 56], [62, 43, -38, 87, -64, -25, -57, 66, -2, 0, 10, -44, 14, -28, -73, -66, 23, 10, -61, -64, -36, 97, -18, 50, -32, -83, -67, -71, -63, 78, -38, -72, -19, -92, -40, -56, 60, -54, -87, 35, 59, -78, 24, -81, -29, -6, -19, -15, -19, -41, -3, 19, 37, 39, 6, -59, 38, -7, -31, -11, 95, -46, 58, 11, -38, -56, 17, -60, 37, -96, -24, -54, 36, 11, -73, 99, 60, -93, -57, -81, 31, 98, -68, 72, -50, 44, 40, 23, -22, 94, 5, -47, 21, -69, -24, -44, 23, -41, 20, -33], [-58, -24, -9, -78, -83, 39, -18, 9, -33, 10, 0, 17, -7, -75, 31, -87, 48, -83, -59, -55, -17, -98, 96, -30, -5, 70, -91, 26, 47, 14, -41, -40, 89, -37, -5, -52, -76, 0, 84, -9, 15, -8, 8, 81, -20, -86, -59, -37, -66, 63, 93, -3, 71, -30, -9, 80, 57, 10, -49, -99, -10, 1, 90, 7, -91, -4, 87, -2, 69, 87, 5, 19, 53, 53, -62, 93, -64, 17, 97, -30, 83, -68, 71, -34, 23, -57, -23, 0, -91, -52, -7, -3, 58, -41, -47, 72, -94, 67, -95, 8], [9, 36, 86, 27, 58, 36, 89, -98, 72, -44, 17, 0, -39, 11, -61, -31, -89, -20, 67, -97, 78, -49, 27, 49, -60, -66, 44, -33, -25, 12, 38, 66, -64, -65, -12, 49, 3, 41, 1, 12, -22, -24, -53, 81, 41, -10, -37, 66, 12, 45, 86, 82, -92, -21, -38, 84, 10, 57, 62, -41, -20, 37, 38, -93, 32, -69, -16, -2, -27, -71, -23, 0, -27, 62, 6, -99, 99, -49, -2, 31, -9, -64, -29, 37, -39, -88, 97, 39, -39, 64, -95, -49, -53, 95, -15, -99, -76, -37, 82, -57], [-66, -11, -93, 83, 93, -17, 65, -87, 2, 14, -7, -39, 0, -46, -33, -55, -68, 57, 28, -94, -90, -69, -83, -82, 60, -36, -91, 12, 62, -65, 44, 30, 6, 26, 99, 0, -45, 79, -40, 76, -54, -26, -30, 36, -34, -89, 70, 7, 15, -4, 45, 58, -78, -32, 48, -55, 17, 30, 55, -49, -97, 44, -81, -99, 23, -4, 27, -47, -67, -68, -38, 89, -21, -36, 24, 8, -63, -28, 61, -32, 57, 89, -37, -69, -49, -58, 86, 48, -96, 60, -100, 28, -48, 72, -56, 46, -62, 6, -32, -92], [91, 55, -59, 6, 88, 37, 6, 51, -32, -28, -75, 11, -46, 0, -80, -72, -18, -95, -54, -57, 76, -3, 30, -77, 49, -5, 98, -74, 21, 33, -16, 25, -41, 50, 40, 65, 44, 78, 34, 45, 34, 89, 41, -9, 10, 88, 41, 51, -97, -46, 46, 93, -27, 85, 14, 29, -94, -17, -28, -17, -82, -78, -79, 93, 62, 89, -5, 94, -50, -39, -68, 39, 65, -14, -20, 42, -75, -19, 2, -67, -61, -87, 9, 62, 80, 55, -29, 71, 49, 0, 48, 64, -39, 23, 17, -8, -38, 37, -74, 60], [9, -25, 75, 18, 97, 41, -8, -49, 23, -73, 31, -61, -33, -80, 0, 8, -11, -44, -86, 17, -41, 36, 20, -89, 17, -72, 67, -35, 92, -11, 57, -82, -16, 31, 11, -86, 31, 3, 35, -70, 34, -19, -43, -2, -15, 46, -64, -20, -39, 78, 27, 79, 55, 71, 47, -43, 58, -15, -26, -37, -2, -100, -46, 42, -94, 86, -34, -60, -36, 86, 51, 88, -99, 71, -86, -19, 23, -93, 44, -81, 84, 21, 26, -42, 12, -98, 7, 51, -7, 66, 77, 87, 79, 31, 20, 89, 12, 27, -65, 45], [86, -70, -22, 39, -72, 83, -81, -56, 22, -66, -87, -31, -55, -72, 8, 0, -5, 86, 81, -67, 47, 47, 49, -41, -82, 21, -17, -74, -18, 40, 16, -47, -73, -64, -29, 12, -41, -30, -37, -60, 3, -65, -50, 36, 64, 49, -76, 94, 12, -65, -85, -99, 14, -66, -1, -47, 27, 20, -44, 41, -50, 8, 75, 19, 21, -36, 63, 69, -62, 36, 78, -25, -46, 46, -46, 4, -55, 74, -59, 86, -18, 30, 87, 63, 16, 77, -93, -4, -43, 93, -79, -42, 38, 0, 55, 6, -70, 19, -3, 83], [-47, 55, -35, -73, -4, 30, -34, 70, -43, 23, 48, -89, -68, -18, -11, -5, 0, -10, 12, 59, 73, -41, 50, -33, 72, 75, 40, 44, 85, 33, 79, -48, -61, 77, 36, 34, 23, -89, 18, 82, 7, -55, 47, -69, 81, -34, -42, -10, -3, 35, 95, 41, -71, -80, -96, 39, 33, 88, 60, 93, 31, -33, -25, -19, -58, -44, -56, -69, 5, 33, -96, -94, 12, -38, -28, -71, 65, -43, 93, 61, -52, 52, -12, -24, 76, 75, -38, -35, 91, 44, -45, 38, -36, 51, 61, -19, 17, 8, 95, 32], [-28, 70, -55, -39, 19, 91, 0, -21, 59, 10, -83, -20, 57, -95, -44, 86, -10, 0, -61, 45, 74, -34, 92, 91, -63, -91, 61, 68, -97, -94, -73, 48, 38, -25, 42, 36, 43, 75, 94, 55, -84, 67, -86, -64, -84, -23, 46, -2, 89, -87, 15, -20, -84, -67, -44, 4, 7, -95, 13, -76, 11, 78, -9, -19, -29, 84, -85, -65, 36, 23, -8, 90, -69, 10, 22, 72, -99, -93, 72, 57, -8, -8, -45, -21, 38, 39, -37, -48, 23, 16, -34, -71, -68, -58, 99, -78, 31, -56, 97, -73], [-74, -93, -76, 5, -32, 8, -71, 99, 41, -61, -59, 67, 28, -54, -86, 81, 12, -61, 0, 93, 12, -61, 76, 65, 41, -58, 64, -54, 14, 86, 90, 97, 74, -66, 67, 49, -16, 7, -24, -3, -86, -12, 34, 10, 81, 46, -58, -62, -10, -26, 77, -95, 98, -16, 99, 13, 3, -18, -92, 21, 84, 97, -79, -20, 70, -2, -94, 34, -72, -80, -85, -3, -67, -95, -75, 22, -61, -37, -85, -51, -90, 0, -46, -32, -33, -91, 7, 47, 81, -27, 0, -38, -87, -41, 36, -42, 16, -94, 89, -58], [13, 94, -13, -44, 4, 61, -2, 29, 31, -64, -55, -97, -94, -57, 17, -67, 59, 45, 93, 0, -91, -51, -5, 27, 99, -72, 65, 43, -13, 12, 51, 57, 66, -66, -24, 85, 5, -17, 16, -90, 88, 12, 21, 71, -83, 83, 12, -92, 33, 65, 1, 90, 86, 74, 77, 70, 6, 98, 82, 67, 0, 23, 52, 38, 39, -68, 49, -87, 21, 27, -90, -2, 90, -29, -13, -28, 45, -98, 63, 44, 33, -62, 21, -2, 10, 69, 57, -73, 71, 10, 87, 17, 21, -12, -79, 31, 86, -95, 87, -40], [-80, 57, -42, -88, -80, 51, 87, -56, -91, -36, -17, 78, -90, 76, -41, 47, 73, 74, 12, -91, 0, -92, 85, -6, 70, 25, -90, 96, -40, 55, 77, 47, 46, 61, 79, 79, 49, 59, -78, 2, -87, -52, -76, -91, 24, 12, -4, 59, -11, -46, -9, 95, 88, 58, 62, 81, 5, -70, -59, 12, 28, 13, 1, -70, 61, 93, -81, 42, 47, 16, 43, -43, -40, 99, 97, -1, 4, -17, -78, 17, 59, 99, -14, -61, -19, 4, -19, 98, 90, -61, 81, -63, -43, 69, -7, -30, -33, 96, -58, 60], [34, -39, 67, -78, -51, -51, 9, 12, 59, 97, -98, -49, -69, -3, 36, 47, -41, -34, -61, -51, -92, 0, -57, 45, 66, 24, 5, -61, -61, 35, -14, 6, 35, -53, 45, -74, -62, 42, -35, -70, 50, -82, -72, -79, 37, -6, -77, 46, -23, 99, -56, -33, 31, -33, -57, 19, 83, 50, 13, 67, -61, 37, 99, 57, -100, 80, -10, -68, -18, 23, -38, -21, -65, 20, -3, -7, 3, 7, 3, 89, -46, -10, 60, -87, -9, -67, -79, -49, 27, 36, -69, 12, -17, -39, -36, -18, -78, -93, -97, -25], [56, -29, 72, 12, -59, -31, 8, -10, 3, -18, 96, 27, -83, 30, 20, 49, 50, 92, 76, -5, 85, -57, 0, -50, 31, 25, -68, 70, 89, 28, -16, -63, 57, 30, -30, 56, -32, -76, 96, -54, 42, 26, -78, 19, -75, -50, -76, -23, 74, -63, -71, -62, 72, 63, -10, 38, -41, -73, 42, -65, -28, 52, -30, -85, 12, 64, 91, 40, 21, -49, 31, -58, -35, -16, 70, -79, -27, -19, 97, -9, -56, -8, 93, 2, 84, 14, -51, -45, 5, 60, -16, -23, 41, 38, 87, -40, 61, -78, 86, 91], [-57, -76, -13, -74, -68, 73, -34, 61, -58, 50, -30, 49, -82, -77, -89, -41, -33, 91, 65, 27, -6, 45, -50, 0, 58, -32, -34, -89, -74, -20, 24, 55, -80, -72, 80, 60, 18, -31, -30, -90, 62, -32, 49, 6, -66, 88, 35, -59, -7, -68, 21, 57, -86, 7, 54, -27, 82, 94, -3, -13, -66, 24, -84, 93, 55, 11, 46, -4, 5, -92, 43, 64, 21, -78, -59, -93, 57, -50, 7, 80, -75, 71, -85, 94, -4, 19, 80, -69, 44, -77, -51, 61, -7, -64, 19, 36, 62, -30, -31, 10], [54, 85, 8, 44, -28, 2, 59, 82, 36, -32, -5, -60, 60, 49, 17, -82, 72, -63, 41, 99, 70, 66, 31, 58, 0, 85, -51, 74, -21, -52, 70, 80, 8, -67, -42, 87, -78, 5, -100, 16, -34, 57, 65, -11, 88, 37, -85, -87, -20, -18, -2, 73, 37, -1, -33, -97, 91, 84, -97, 40, -51, -26, -23, -59, -97, -51, -56, -95, -8, -28, 93, -8, 69, -36, 71, -91, 1, -69, 62, -78, -2, 90, -47, 54, 91, 62, -64, -44, -37, 79, -30, 17, -51, -83, -13, -31, 46, -87, 68, 72], [46, 88, 8, -84, 9, -82, -97, -11, 56, -83, 70, -66, -36, -5, -72, 21, 75, -91, -58, -72, 25, 24, 25, -32, 85, 0, -87, 30, -79, -73, 60, -5, -35, -3, 24, 55, 43, -2, -17, -91, 80, 15, -3, -15, 76, -54, -79, 72, 60, -34, 61, 93, 51, 46, 73, 45, 97, -95, 42, -40, 96, 28, 43, 55, -66, 17, -81, 17, 29, -92, -99, 5, -89, 78, -82, -87, -37, 43, 7, 11, -61, -71, 65, -38, 80, -59, -38, 2, -63, 93, -6, -2, -32, 60, -51, 57, 47, -81, -44, -36], [54, 17, -12, 21, 17, -82, -85, -30, -43, -67, -91, 44, -91, 98, 67, -17, 40, 61, 64, 65, -90, 5, -68, -34, -51, -87, 0, -54, -13, 14, 88, 58, -52, 46, 92, 86, 73, 5, -93, 39, 37, 67, 22, 1, -22, 10, 62, -9, -51, 12, -98, -29, 78, -36, -86, 22, -28, 27, 55, -88, -11, -70, 63, -58, -12, -74, -10, 60, 88, 80, -73, -33, 0, 87, -46, -3, -82, -92, 68, -26, 99, -1, -62, -55, -50, -89, 23, -7, 19, -80, 75, -27, 22, -36, -18, 37, 78, -79, -32, -72], [56, -98, -21, -53, 79, -77, 39, 28, 54, -71, 26, -33, 12, -74, -35, -74, 44, 68, -54, 43, 96, -61, 70, -89, 74, 30, -54, 0, -37, 0, -6, -95, 66, -17, 32, 97, -95, -76, 79, -85, -75, -43, -23, 30, 74, -54, -47, 6, -68, -34, -28, 43, -97, -92, 80, 11, 53, -16, -87, 15, 11, 39, -49, -5, 36, -29, 24, 38, 38, -39, 29, -33, 88, -20, 53, -46, -50, 70, 82, 25, 30, -52, -31, 18, -52, 78, 78, 39, -99, 4, 7, 81, -54, -11, 23, -60, -92, -71, -49, -30], [-36, 64, -81, -9, 54, -23, 63, 4, -16, -63, 47, -25, 62, 21, 92, -18, 85, -97, 14, -13, -40, -61, 89, -74, -21, -79, -13, -37, 0, -55, 55, -49, 27, 65, -36, -52, 37, -5, 63, -71, 68, -61, 38, 83, 48, -69, 94, 78, 37, -28, 16, 13, 35, 73, 10, -40, 64, 64, -60, -11, -51, -91, -81, 49, 18, 91, 45, -31, 62, -66, -90, -2, 61, -94, 75, -90, -45, 15, -86, 19, 53, -43, -9, -72, -44, 14, 75, 68, -20, 71, -61, -64, 60, -1, -75, -54, -25, -78, -3, -19], [1, -65, 92, -59, 78, 55, 55, 56, 96, 78, 14, 12, -65, 33, -11, 40, 33, -94, 86, 12, 55, 35, 28, -20, -52, -73, 14, 0, -55, 0, 59, -60, 5, 58, -83, 27, 59, 93, 57, 50, -20, -94, -28, 52, 8, 40, 89, -31, -55, 66, 61, 65, 82, -83, -79, -27, -47, -94, 39, 59, -71, -63, 28, -59, -31, 70, 8, -48, -69, 64, -86, -84, 56, 19, -17, -81, 42, -81, -62, -19, 78, 11, -71, -74, 5, -55, -1, 48, 39, -15, 42, 38, 0, 14, -1, -29, -57, 46, -37, -19], [-3, 31, 32, 68, 85, -31, -73, -18, 23, -38, -41, 38, 44, -16, 57, 16, 79, -73, 90, 51, 77, -14, -16, 24, 70, 60, 88, -6, 55, 59, 0, -98, -31, -35, -89, -87, -48, -46, 2, 95, 9, -74, 91, -78, -71, 69, 56, 79, -61, 55, -72, -88, -6, -61, -51, -98, 65, 44, -29, 25, 46, -9, -27, -72, 54, 51, 96, -99, -61, 30, 48, -99, 14, -15, -72, 85, 29, -3, 68, -8, -29, 13, -87, 84, 58, 38, 72, -100, -68, -1, 78, -55, -99, -73, 39, -75, 51, -26, 98, -37], [-97, -67, -71, 38, 66, -39, 94, 83, 40, -72, -40, 66, 30, 25, -82, -47, -48, 48, 97, 57, 47, 6, -63, 55, 80, -5, 58, -95, -49, -60, -98, 0, -54, -38, -47, -51, 89, 18, -7, -64, 96, 0, -55, 6, -73, -70, -37, 18, 63, -13, 27, 21, -62, -92, 92, 34, 7, 48, -72, 12, 45, -86, -48, -36, 13, 18, -53, 31, -95, -36, 47, -36, -59, -42, 47, 38, 52, 5, 34, 96, -29, 31, 63, -54, -6, -9, -79, 81, 38, 77, 55, 9, -10, 27, 99, 35, 41, 92, -72, 54], [-46, -8, 60, 81, 32, -89, -39, -51, 30, -19, 89, -64, 6, -41, -16, -73, -61, 38, 74, 66, 46, 35, 57, -80, 8, -35, -52, 66, 27, 5, -31, -54, 0, 37, 43, 47, -1, 67, 87, -30, -16, 10, -68, -12, 81, 81, 23, 0, 83, 90, 61, 41, -4, -88, 22, 76, 41, 87, -99, -69, -53, 19, 30, -5, 51, -30, -9, -75, 54, 9, 43, -4, 44, -76, 36, 68, -36, -27, 23, 48, 76, -90, 53, 6, 40, 17, -27, 6, 54, -56, -67, -50, -70, 49, -13, 9, 73, 10, -21, 77], [47, -94, 72, 58, -34, -27, -21, -2, 75, -92, -37, -65, 26, 50, 31, -64, 77, -25, -66, -66, 61, -53, 30, -72, -67, -3, 46, -17, 65, 58, -35, -38, 37, 0, 84, -58, -8, -1, 35, -83, 57, 81, 51, -83, -51, -81, 54, 86, -10, -81, 94, 75, 97, 84, -49, -34, -51, -99, -54, 9, -78, -65, 55, 70, 41, -55, -26, -34, -1, 41, 48, 85, 98, 35, 15, -89, 35, -32, 27, 35, -29, 92, 1, 99, 4, 30, 52, 12, 94, 11, -72, 99, -99, -14, 23, 37, -62, 35, 90, 69], [71, -2, -52, 63, 97, -71, -80, 36, 0, -40, -5, -12, 99, 40, 11, -29, 36, 42, 67, -24, 79, 45, -30, 80, -42, 24, 92, 32, -36, -83, -89, -47, 43, 84, 0, 65, -48, 22, -89, -95, -32, 80, 49, 8, 20, 98, -69, 31, -17, -83, -6, 68, 98, 88, -91, -74, 22, -81, -73, -65, -62, 23, 27, -48, 62, -78, -38, -74, -80, -68, 43, 98, -82, 29, -6, 70, -82, -20, 33, -94, -27, -68, 41, -17, -3, 65, -24, -97, 0, 52, 30, 81, -19, -82, -32, -46, -66, 87, 84, 1], [-68, 42, 90, -79, -93, -34, -73, 76, 56, -56, -52, 49, 0, 65, -86, 12, 34, 36, 49, 85, 79, -74, 56, 60, 87, 55, 86, 97, -52, 27, -87, -51, 47, -58, 65, 0, -51, -16, -19, 54, 16, 7, 39, 89, 77, -23, 38, -59, 69, 69, -5, 41, -22, -21, 42, -27, -100, 98, -83, 58, 52, -1, -36, 37, 66, 6, 8, 23, 77, -6, 93, 96, 71, -88, -79, -93, -21, 18, -22, -37, -43, -75, -19, -3, 69, -4, 19, -62, -94, 80, -4, -24, 78, -37, -90, -45, 89, 93, 32, -74], [52, -73, -69, 49, -69, 2, -63, -56, -19, 60, -76, 3, -45, 44, 31, -41, 23, 43, -16, 5, 49, -62, -32, 18, -78, 43, 73, -95, 37, 59, -48, 89, -1, -8, -48, -51, 0, -68, 63, 97, -42, 15, 85, -29, 2, -27, -2, -27, 22, -94, -66, -62, -66, 22, 7, -67, 90, 85, 21, 75, -61, 44, -44, -94, -20, 3, 60, 3, -60, 28, -5, 85, -59, 13, 18, -65, 30, -66, -2, -1, 46, 45, -38, -94, -92, 17, 83, -71, -42, -67, -63, -29, 24, -86, -3, 18, -83, 62, 55, -49], [-80, 66, 33, 90, 10, -90, -69, -16, 71, -54, 0, 41, 79, 78, 3, -30, -89, 75, 7, -17, 59, 42, -76, -31, 5, -2, 5, -76, -5, 93, -46, 18, 67, -1, 22, -16, -68, 0, -87, -14, 0, 76, -36, -61, -55, 97, 59, -66, -23, -22, -62, -34, -91, 65, 34, 98, 91, 87, 1, -75, 89, 86, -30, 42, 95, 68, -17, 46, -93, 84, -76, 65, 97, 81, 18, -82, -22, -48, 13, 36, 35, -33, 88, -21, 66, -95, 92, 10, -61, -47, 83, -78, -94, -51, -12, 30, 86, 8, 81, -36], [-97, -96, -18, -80, -40, 41, -84, 43, 94, -87, 84, 1, -40, 34, 35, -37, 18, 94, -24, 16, -78, -35, 96, -30, -100, -17, -93, 79, 63, 57, 2, -7, 87, 35, -89, -19, 63, -87, 0, -11, -15, -82, -62, 16, 44, 69, 38, -11, 28, -67, 49, -99, 44, -68, -38, -39, 26, 65, 24, -81, 39, 24, 68, -70, -71, -8, 10, -85, -99, 2, -85, -54, -99, 73, -84, 89, 34, -13, 72, 24, -25, 41, 78, 10, -64, -8, -79, 94, -27, -1, 21, -41, -9, -40, -78, 65, 95, 9, -71, 95], [14, -99, -76, 28, 57, -84, 94, 64, -59, 35, -9, 12, 76, 45, -70, -60, 82, 55, -3, -90, 2, -70, -54, -90, 16, -91, 39, -85, -71, 50, 95, -64, -30, -83, -95, 54, 97, -14, -11, 0, -28, -71, 20, 19, 58, 11, 95, 4, -88, 30, -83, 66, -55, -56, -58, 26, 54, 84, -88, -48, 57, 99, 65, 91, -83, -11, -92, 65, -32, 37, -56, 63, 98, 13, -91, 97, -9, 27, -85, 77, -33, 53, 15, -28, 82, -94, 33, -6, 4, 7, -57, 45, 80, 54, 55, -39, 9, -63, -1, 36], [-48, -6, 19, -32, -94, -59, -2, -36, 73, 59, 15, -22, -54, 34, 34, 3, 7, -84, -86, 88, -87, 50, 42, 62, -34, 80, 37, -75, 68, -20, 9, 96, -16, 57, -32, 16, -42, 0, -15, -28, 0, -34, 74, -23, 8, -49, 17, 81, -33, 16, -85, 20, 26, -100, -69, -1, 26, 65, 14, -89, 81, 9, 1, 64, 18, -27, -70, 13, 95, 70, -16, 25, -88, -4, 85, 1, 43, 7, -65, -19, -34, 16, 45, 73, 63, -95, 14, 63, 37, -86, 31, -93, 29, 15, 8, -94, -50, -76, -90, 54], [-81, -81, -59, -41, 39, 10, -1, -27, -21, -78, -8, -24, -26, 89, -19, -65, -55, 67, -12, 12, -52, -82, 26, -32, 57, 15, 67, -43, -61, -94, -74, 0, 10, 81, 80, 7, 15, 76, -82, -71, -34, 0, -33, -13, -17, -33, -68, -57, 13, 30, 90, -9, -38, 77, 96, -68, 60, 35, -89, -21, 71, -12, 33, -52, -53, 85, 11, -81, -51, 53, 1, 96, -86, 61, 33, 6, 48, 11, 57, -72, -80, 66, 34, -87, 93, 26, -30, 64, 28, 40, 6, -42, 97, -9, 70, 19, -81, 18, -70, -72], [71, -67, 62, -70, -36, -31, 52, 55, -39, 24, 8, -53, -30, 41, -43, -50, 47, -86, 34, 21, -76, -72, -78, 49, 65, -3, 22, -23, 38, -28, 91, -55, -68, 51, 49, 39, 85, -36, -62, 20, 74, -33, 0, 12, 9, 99, 41, -9, -86, 0, -95, -80, 44, -22, -57, 27, 71, 1, 57, 6, 18, -16, -61, 51, -34, -69, -79, 60, -97, 74, 81, -5, -33, 1, -49, 88, -36, -34, 52, 55, -49, 14, 15, 35, -78, 92, 18, -15, 88, 80, 92, -51, -3, 32, -7, 38, -50, -98, -62, -48], [-25, -56, 14, -68, 56, -99, -70, 88, -100, -81, 81, 81, 36, -9, -2, 36, -69, -64, 10, 71, -91, -79, 19, 6, -11, -15, 1, 30, 83, 52, -78, 6, -12, -83, 8, 89, -29, -61, 16, 19, -23, -13, 12, 0, 89, 48, -56, -69, 66, -25, 19, -68, 90, -42, 71, 48, 35, -9, -99, 60, 97, -27, -22, -72, 53, -31, -87, 11, -62, 15, -77, -49, 51, 46, 29, -87, -1, 10, -50, 86, -68, -37, -56, 29, 73, 98, -12, -91, 22, 64, -35, 95, 26, 54, 12, -61, -90, 43, -28, 27], [33, 5, -87, -90, -74, -82, -60, 8, -35, -29, -20, 41, -34, 10, -15, 64, 81, -84, 81, -83, 24, 37, -75, -66, 88, 76, -22, 74, 48, 8, -71, -73, 81, -51, 20, 77, 2, -55, 44, 58, 8, -17, 9, 89, 0, -98, -39, -42, 58, 66, 37, 11, 99, -60, 83, 86, 34, 19, -1, 56, 57, -92, -11, 40, 64, -91, 51, 99, 97, -66, -7, -50, -93, 48, -31, 85, 99, 74, -100, -35, -84, 5, 98, -25, -100, -77, 60, -81, 1, 83, -44, -98, -77, -69, 34, -35, -20, -58, 83, 31], [36, -50, -65, -69, -63, -78, -14, -2, 67, -6, -86, -10, -89, 88, 46, 49, -34, -23, 46, 83, 12, -6, -50, 88, 37, -54, 10, -54, -69, 40, 69, -70, 81, -81, 98, -23, -27, 97, 69, 11, -49, -33, 99, 48, -98, 0, 8, 43, 89, -1, 85, -74, 16, 17, 1, 12, -78, 0, 33, 54, 27, -46, -39, 81, -96, -52, 86, 55, -29, -79, -86, -52, -97, 31, -66, 18, -95, 26, -76, 62, -64, -75, -9, -12, -79, -10, 60, -4, 45, 99, 75, -37, 97, 27, 28, 6, -4, 94, -10, 64], [62, 49, 5, -64, 3, 85, -72, -65, -57, -19, -59, -37, 70, 41, -64, -76, -42, 46, -58, 12, -4, -77, -76, 35, -85, -79, 62, -47, 94, 89, 56, -37, 23, 54, -69, 38, -2, 59, 38, 95, 17, -68, 41, -56, -39, 8, 0, -85, 49, 7, -67, 88, -29, -56, 37, 10, -65, -95, 80, 38, 66, 32, -96, -85, -62, -37, -98, 2, 9, 27, -73, -95, 0, -72, 82, -56, 22, 46, 44, 95, -25, 89, 45, -42, -47, -84, -57, 4, 88, 64, 71, 80, 26, -24, 8, -77, 88, 83, -58, 49], [98, -88, 68, -29, 83, -10, -37, 47, -13, -15, -37, 66, 7, 51, -20, 94, -10, -2, -62, -92, 59, 46, -23, -59, -87, 72, -9, 6, 78, -31, 79, 18, 0, 86, 31, -59, -27, -66, -11, 4, 81, -57, -9, -69, -42, 43, -85, 0, 13, -5, -70, -54, -97, -28, -7, 76, -73, 76, -8, -36, 56, 13, 50, 92, -85, 38, -16, 93, 0, -4, 30, -73, -15, 35, 54, 23, -24, 56, -6, 2, 52, -76, -86, -59, 59, 64, 50, -41, 68, 33, 93, 77, -93, -61, 58, -72, -17, -59, 77, -15], [91, -46, -94, 64, 72, 92, 92, 64, -30, -19, -66, 12, 15, -97, -39, 12, -3, 89, -10, 33, -11, -23, 74, -7, -20, 60, -51, -68, 37, -55, -61, 63, 83, -10, -17, 69, 22, -23, 28, -88, -33, 13, -86, 66, 58, 89, 49, 13, 0, -35, 48, -4, -38, 79, 31, 1, 14, -90, -77, 82, -94, 13, -44, -91, 60, -59, 14, 22, 58, -82, -37, -33, 40, 88, -20, 71, -74, 30, 99, -55, -1, -34, 66, 26, 73, -42, 88, 61, -84, -20, 87, -52, 60, 1, -23, 58, -36, -6, -5, -74], [-68, -21, 79, -95, 94, 28, -46, 45, -74, -41, 63, 45, -4, -46, 78, -65, 35, -87, -26, 65, -46, 99, -63, -68, -18, -34, 12, -34, -28, 66, 55, -13, 90, -81, -83, 69, -94, -22, -67, 30, 16, 30, 0, -25, 66, -1, 7, -5, -35, 0, 75, 69, -87, -31, 45, -51, 30, 95, -84, -31, -57, -50, 2, -63, -50, 88, -74, 97, -82, 74, -33, -10, -99, -83, 4, 33, -82, 21, 32, -64, -53, -56, 20, -78, 75, -90, 79, -73, -81, 44, -23, 69, -49, -53, -13, 54, 70, 62, -45, -14], [30, 57, -68, -23, 13, -87, -56, 64, 7, -3, 93, 86, 45, 46, 27, -85, 95, 15, 77, 1, -9, -56, -71, 21, -2, 61, -98, -28, 16, 61, -72, 27, 61, 94, -6, -5, -66, -62, 49, -83, -85, 90, -95, 19, 37, 85, -67, -70, 48, 75, 0, 81, -28, -30, -32, 48, -28, 6, 4, -55, 14, 21, -28, 43, -3, -39, 17, 25, -38, -34, -82, -42, -17, -14, -31, 44, 5, 17, -20, -18, -70, 57, 29, -62, -12, -19, 81, 91, 15, 44, 99, -11, 28, 59, 56, 38, -48, -80, -40, -40], [-22, 83, -59, 14, -40, -70, -18, -17, 91, 19, -3, 82, 58, 93, 79, -99, 41, -20, -95, 90, 95, -33, -62, 57, 73, 93, -29, 43, 13, 65, -88, 21, 41, 75, 68, 41, -62, -34, -99, 66, 20, -9, -80, -68, 11, -74, 88, -54, -4, 69, 81, 0, -44, 36, 79, 30, 61, -69, 28, 49, -22, -49, 60, -72, -53, 11, 22, 75, 12, -3, 77, 29, -26, -59, 15, 80, -82, -39, 88, 48, -60, 4, -93, 1, 21, 98, -24, 89, 34, -54, -44, 66, 61, -93, 39, 88, -56, 72, -41, 30], [30, -8, -48, 39, -79, -14, -75, 29, 70, 37, 71, -92, -78, -27, 55, 14, -71, -84, 98, 86, 88, 31, 72, -86, 37, 51, 78, -97, 35, 82, -6, -62, -4, 97, 98, -22, -66, -91, 44, -55, 26, -38, 44, 90, 99, 16, -29, -97, -38, -87, -28, -44, 0, -6, -81, 16, 6, -74, -82, 74, 42, -68, 69, 41, -85, 5, 99, -82, -87, 89, -8, 30, 41, 85, 1, -80, -1, -16, 70, 55, -70, -15, -18, -78, 85, 65, -31, -16, 92, 17, -15, -30, 91, -55, -79, -7, 95, -61, -6, 93], [31, -23, -51, -30, 6, -63, -29, -100, 15, 39, -30, -21, -32, 85, 71, -66, -80, -67, -16, 74, 58, -33, 63, 7, -1, 46, -36, -92, 73, -83, -61, -92, -88, 84, 88, -21, 22, 65, -68, -56, -100, 77, -22, -42, -60, 17, -56, -28, 79, -31, -30, 36, -6, 0, 43, -91, 86, -25, 30, 70, -53, -39, -7, 81, -60, 55, 56, 35, -28, -86, -96, -53, 67, -29, -31, 68, -6, 86, -70, 93, -95, 69, 29, -35, 96, -100, 69, -88, -90, 96, 87, 35, -40, -14, 4, -21, 80, 98, -80, -89], [-2, 26, 17, -20, -81, -41, 59, -50, 89, 6, -9, -38, 48, 14, 47, -1, -96, -44, 99, 77, 62, -57, -10, 54, -33, 73, -86, 80, 10, -79, -51, 92, 22, -49, -91, 42, 7, 34, -38, -58, -69, 96, -57, 71, 83, 1, 37, -7, 31, 45, -32, 79, -81, 43, 0, 31, 88, -38, -64, 97, 2, 60, 99, 7, 70, -55, -16, 96, -88, 7, -48, -12, -82, -27, -42, -23, 39, -85, 25, -47, -63, -64, 44, 80, 38, 94, -54, -22, 33, 72, 53, 59, -100, -27, 65, -90, 50, -24, 7, 89], [-32, -31, -60, 23, -48, -17, 92, 63, 22, -59, 80, 84, -55, 29, -43, -47, 39, 4, 13, 70, 81, 19, 38, -27, -97, 45, 22, 11, -40, -27, -98, 34, 76, -34, -74, -27, -67, 98, -39, 26, -1, -68, 27, 48, 86, 12, 10, 76, 1, -51, 48, 30, 16, -91, 31, 0, 72, -95, -74, 57, -73, 89, -46, -79, -54, 86, 76, -68, -98, 94, 43, -7, -74, 90, 23, 94, 78, 94, 38, 53, 50, -2, -86, 57, 5, -30, -46, 12, 87, -69, -20, -48, -48, 13, 57, -26, -51, -25, 14, -73], [15, -52, -56, 80, -74, 71, -77, -43, -53, 38, 57, 10, 17, -94, 58, 27, 33, 7, 3, 6, 5, 83, -41, 82, 91, 97, -28, 53, 64, -47, 65, 7, 41, -51, 22, -100, 90, 91, 26, 54, 26, 60, 71, 35, 34, -78, -65, -73, 14, 30, -28, 61, 6, 86, 88, 72, 0, -61, 86, -54, -30, -72, 14, 13, -44, 95, 33, -73, -3, 5, -37, 27, 39, 2, -98, 55, -2, -76, 11, -28, -6, 76, 94, -7, -98, 91, 21, -54, 66, 40, 80, 27, -37, 92, 95, 65, 60, 71, 1, 63], [3, -33, 72, 33, 57, 26, -96, -8, 28, -7, 10, 57, 30, -17, -15, 20, 88, -95, -18, 98, -70, 50, -73, 94, 84, -95, 27, -16, 64, -94, 44, 48, 87, -99, -81, 98, 85, 87, 65, 84, 65, 35, 1, -9, 19, 0, -95, 76, -90, 95, 6, -69, -74, -25, -38, -95, -61, 0, -22, 81, -22, 34, -35, -61, 95, 97, 87, 86, -92, -21, 53, -90, -45, -25, 42, 40, 31, -12, -36, 72, -50, -52, -21, -66, 97, -68, -50, 69, -7, -37, 13, 63, -6, 10, -73, 84, 80, 0, 57, -37], [-32, -8, 55, 69, -45, 26, 28, -1, -29, -31, -49, 62, 55, -28, -26, -44, 60, 13, -92, 82, -59, 13, 42, -3, -97, 42, 55, -87, -60, 39, -29, -72, -99, -54, -73, -83, 21, 1, 24, -88, 14, -89, 57, -99, -1, 33, 80, -8, -77, -84, 4, 28, -82, 30, -64, -74, 86, -22, 0, -1, 28, -15, -45, -90, 4, 34, 60, -62, 74, 30, 81, -61, 34, 27, 17, -35, 37, -16, 32, -1, 77, -45, 4, -10, -35, -27, -100, -84, -63, 47, 83, -82, -86, 1, -13, -37, 91, -80, 91, -36], [66, 33, -18, 59, -98, -4, 76, 26, 44, -11, -99, -41, -49, -17, -37, 41, 93, -76, 21, 67, 12, 67, -65, -13, 40, -40, -88, 15, -11, 59, 25, 12, -69, 9, -65, 58, 75, -75, -81, -48, -89, -21, 6, 60, 56, 54, 38, -36, 82, -31, -55, 49, 74, 70, 97, 57, -54, 81, -1, 0, 48, -35, 87, 2, -4, -8, -30, 25, 13, 77, -72, -13, 17, 51, 34, 30, 59, -63, -68, 59, -72, -75, -60, -2, -11, -28, -47, 11, 4, 74, -42, 61, 28, -83, -87, -94, 43, 13, 63, 29], [-9, 61, 42, -44, 83, -4, -66, -75, -41, 95, -10, -20, -97, -82, -2, -50, 31, 11, 84, 0, 28, -61, -28, -66, -51, 96, -11, 11, -51, -71, 46, 45, -53, -78, -62, 52, -61, 89, 39, 57, 81, 71, 18, 97, 57, 27, 66, 56, -94, -57, 14, -22, 42, -53, 2, -73, -30, -22, 28, 48, 0, -22, 39, 31, -6, -29, -40, -91, -54, -56, 91, -80, -74, -68, 81, -39, -29, -43, -98, -15, -100, -18, -82, 85, -68, 2, 66, -91, -55, 73, -3, -89, -1, 6, -19, 40, 85, 53, 69, -18], [13, -41, 57, 90, -37, -97, 54, 96, 48, -46, 1, 37, 44, -78, -100, 8, -33, 78, 97, 23, 13, 37, 52, 24, -26, 28, -70, 39, -91, -63, -9, -86, 19, -65, 23, -1, 44, 86, 24, 99, 9, -12, -16, -27, -92, -46, 32, 13, 13, -50, 21, -49, -68, -39, 60, 89, -72, 34, -15, -35, -22, 0, -62, -1, -79, 93, -17, -66, -98, -40, 12, 83, -61, 91, 45, -94, 39, -3, 72, -2, -89, -98, -38, 97, -87, -5, 0, 8, 88, -92, 4, 21, 9, 25, 46, -49, -64, -87, -18, 11], [-29, -88, 71, -62, -82, 24, -88, -82, -18, 58, 90, 38, -81, -79, -46, 75, -25, -9, -79, 52, 1, 99, -30, -84, -23, 43, 63, -49, -81, 28, -27, -48, 30, 55, 27, -36, -44, -30, 68, 65, 1, 33, -61, -22, -11, -39, -96, 50, -44, 2, -28, 60, 69, -7, 99, -46, 14, -35, -45, 87, 39, -62, 0, -80, 96, 29, 63, -88, 71, -24, 93, 68, 3, 16, -94, 46, 63, 95, -18, -21, 16, -76, -94, -100, 79, 1, 88, -27, 19, -57, 57, 27, 31, -82, -19, -27, -79, 71, -60, 82], [-43, -94, 9, -95, -9, 11, 47, -58, 54, 11, 7, -93, -99, 93, 42, 19, -19, -19, -20, 38, -70, 57, -85, 93, -59, 55, -58, -5, 49, -59, -72, -36, -5, 70, -48, 37, -94, 42, -70, 91, 64, -52, 51, -72, 40, 81, -85, 92, -91, -63, 43, -72, 41, 81, 7, -79, 13, -61, -90, 2, 31, -1, -80, 0, -81, -47, -86, 91, 36, 64, -36, 3, -67, 83, -93, -74, 61, 13, 63, 77, 41, -58, -49, -81, -63, 43, -50, 28, 82, -55, -98, -94, 55, 22, -79, 63, -27, 94, -67, 68], [39, 77, -95, 55, 80, 38, -21, 35, -13, -38, -91, 32, 23, 62, -94, 21, -58, -29, 70, 39, 61, -100, 12, 55, -97, -66, -12, 36, 18, -31, 54, 13, 51, 41, 62, 66, -20, 95, -71, -83, 18, -53, -34, 53, 64, -96, -62, -85, 60, -50, -3, -53, -85, -60, 70, -54, -44, 95, 4, -4, -6, -79, 96, -81, 0, 48, -32, 44, 61, -10, -47, -50, -56, 85, 17, 94, -45, 88, 99, 3, -86, 48, 81, 95, 66, 5, -42, 34, 84, -96, 82, 20, -82, 54, -99, -63, 40, 66, -40, 46], [-98, -64, -62, -52, -37, -89, -96, 30, 30, -56, -4, -69, -4, 89, 86, -36, -44, 84, -2, -68, 93, 80, 64, 11, -51, 17, -74, -29, 91, 70, 51, 18, -30, -55, -78, 6, 3, 68, -8, -11, -27, 85, -69, -31, -91, -52, -37, 38, -59, 88, -39, 11, 5, 55, -55, 86, 95, 97, 34, -8, -29, 93, 29, -47, 48, 0, -86, -44, 4, -72, 65, 89, 33, 75, 1, -57, 3, 9, -60, 24, -65, -59, 36, 23, 87, -82, 70, 46, -41, 79, -42, 83, -64, 8, 70, 55, -49, -79, 16, 32], [-46, 64, 58, 29, -7, 1, -68, -100, -88, 17, 87, -16, 27, -5, -34, 63, -56, -85, -94, 49, -81, -10, 91, 46, -56, -81, -10, 24, 45, 8, 96, -53, -9, -26, -38, 8, 60, -17, 10, -92, -70, 11, -79, -87, 51, 86, -98, -16, 14, -74, 17, 22, 99, 56, -16, 76, 33, 87, 60, -30, -40, -17, 63, -86, -32, -86, 0, 54, 27, -22, -92, -87, 72, -44, 78, -77, -53, -34, -91, 40, -35, 55, 8, 79, 32, 9, 17, 7, -5, -45, 48, -35, -39, 7, -46, 34, 14, 36, 25, -5], [-67, 3, 20, 82, 24, 61, 24, -91, 29, -60, -2, -2, -47, 94, -60, 69, -69, -65, 34, -87, 42, -68, 40, -4, -95, 17, 60, 38, -31, -48, -99, 31, -75, -34, -74, 23, 3, 46, -85, 65, 13, -81, 60, 11, 99, 55, 2, 93, 22, 97, 25, 75, -82, 35, 96, -68, -73, 86, -62, 25, -91, -66, -88, 91, 44, -44, 54, 0, 54, 85, 26, -23, 96, 15, -78, -17, -91, -34, 64, 51, -19, -97, -34, 75, 2, -23, 0, -37, 34, -25, -58, 66, -49, 57, 27, 43, 12, -72, 71, -74], [-44, 37, 47, -7, 7, -28, 27, -37, -68, 37, 69, -27, -67, -50, -36, -62, 5, 36, -72, 21, 47, -18, 21, 5, -8, 29, 88, 38, 62, -69, -61, -95, 54, -1, -80, 77, -60, -93, -99, -32, 95, -51, -97, -62, 97, -29, 9, 0, 58, -82, -38, 12, -87, -28, -88, -98, -3, -92, 74, 13, -54, -98, 71, 36, 61, 4, 27, 54, 0, 33, -78, -93, 74, 60, 24, -73, -78, -22, 57, -65, 89, -65, 81, -2, 45, 88, -62, 8, -4, -72, -12, 96, -69, -94, -33, -49, -89, -12, -77, 10], [-63, -62, 61, -49, -67, -67, -32, 32, 7, -96, 87, -71, -68, -39, 86, 36, 33, 23, -80, 27, 16, 23, -49, -92, -28, -92, 80, -39, -66, 64, 30, -36, 9, 41, -68, -6, 28, 84, 2, 37, 70, 53, 74, 15, -66, -79, 27, -4, -82, 74, -34, -3, 89, -86, 7, 94, 5, -21, 30, 77, -56, -40, -24, 64, -10, -72, -22, 85, 33, 0, 89, -8, 52, 46, -37, 20, -4, -83, -39, -41, 0, -83, -100, -2, -29, 71, 89, -5, 47, 65, -15, 11, -43, -96, -54, -12, -2, -17, -60, 9], [82, -68, -25, 63, 11, -34, -61, -94, 92, -24, 5, -23, -38, -68, 51, 78, -96, -8, -85, -90, 43, -38, 31, 43, 93, -99, -73, 29, -90, -86, 48, 47, 43, 48, 43, 93, -5, -76, -85, -56, -16, 1, 81, -77, -7, -86, -73, 30, -37, -33, -82, 77, -8, -96, -48, 43, -37, 53, 81, -72, 91, 12, 93, -36, -47, 65, -92, 26, -78, 89, 0, 54, -78, -6, -10, 52, 53, 26, -30, -25, -99, 22, 54, 13, -48, 13, 36, 49, 42, 68, 85, -62, 7, 29, -80, -87, -27, -21, 88, -52], [-4, -86, -79, -67, 59, 51, -91, -76, 38, -54, 19, 0, 89, 39, 88, -25, -94, 90, -3, -2, -43, -21, -58, 64, -8, 5, -33, -33, -2, -84, -99, -36, -4, 85, 98, 96, 85, 65, -54, 63, 25, 96, -5, -49, -50, -52, -95, -73, -33, -10, -42, 29, 30, -53, -12, -7, 27, -90, -61, -13, -80, 83, 68, 3, -50, 89, -87, -23, -93, -8, 54, 0, 43, -3, -88, -21, 95, 55, -27, 87, -90, -99, -1, 48, -78, -14, 86, -93, 54, -49, 22, 36, -5, 47, -3, 28, -49, 46, 35, -60], [-100, 65, 64, -52, 46, -7, 86, -69, -41, 36, 53, -27, -21, 65, -99, -46, 12, -69, -67, 90, -40, -65, -35, 21, 69, -89, 0, 88, 61, 56, 14, -59, 44, 98, -82, 71, -59, 97, -99, 98, -88, -86, -33, 51, -93, -97, 0, -15, 40, -99, -17, -26, 41, 67, -82, -74, 39, -45, 34, 17, -74, -61, 3, -67, -56, 33, 72, 96, 74, 52, -78, 43, 0, 32, 64, 1, -97, -10, 30, -74, -96, -53, -51, 3, 9, -57, -26, -53, -77, -72, 12, 38, 38, 77, -59, 83, 28, 61, -52, -20], [-88, 61, -97, 47, -57, -50, -20, -48, -54, 11, 53, 62, -36, -14, 71, 46, -38, 10, -95, -29, 99, 20, -16, -78, -36, 78, 87, -20, -94, 19, -15, -42, -76, 35, 29, -88, 13, 81, 73, 13, -4, 61, 1, 46, 48, 31, -72, 35, 88, -83, -14, -59, 85, -29, -27, 90, 2, -25, 27, 51, -68, 91, 16, 83, 85, 75, -44, 15, 60, 46, -6, -3, 32, 0, -1, -1, -53, -96, 12, 88, 67, 71, -20, -91, -65, 24, -32, 24, -77, -55, -76, 35, -60, -27, -43, 17, -84, 99, 36, -12], [39, -16, 96, 7, -55, 88, 92, -21, 56, -73, -62, 6, 24, -20, -86, -46, -28, 22, -75, -13, 97, -3, 70, -59, 71, -82, -46, 53, 75, -17, -72, 47, 36, 15, -6, -79, 18, 18, -84, -91, 85, 33, -49, 29, -31, -66, 82, 54, -20, 4, -31, 15, 1, -31, -42, 23, -98, 42, 17, 34, 81, 45, -94, -93, 17, 1, 78, -78, 24, -37, -10, -88, 64, -1, 0, 58, 38, 95, -66, 4, 65, 92, 1, 18, 39, -78, -67, -15, -13, -97, 16, -51, 98, -81, 8, 7, -59, -38, 1, 68], [25, 11, 53, -53, -56, -58, 20, 75, -30, 99, 93, -99, 8, 42, -19, 4, -71, 72, 22, -28, -1, -7, -79, -93, -91, -87, -3, -46, -90, -81, 85, 38, 68, -89, 70, -93, -65, -82, 89, 97, 1, 6, 88, -87, 85, 18, -56, 23, 71, 33, 44, 80, -80, 68, -23, 94, 55, 40, -35, 30, -39, -94, 46, -74, 94, -57, -77, -17, -73, 20, 52, -21, 1, -1, 58, 0, 53, 14, 95, 14, 4, -70, -77, 75, -32, 53, 28, 88, 32, 22, -33, 72, 26, -25, -2, 41, -26, 78, -97, 45], [-76, 57, 10, -36, -35, 68, -38, -32, -14, 60, -64, 99, -63, -75, 23, -55, 65, -99, -61, 45, 4, 3, -27, 57, 1, -37, -82, -50, -45, 42, 29, 52, -36, 35, -82, -21, 30, -22, 34, -9, 43, 48, -36, -1, 99, -95, 22, -24, -74, -82, 5, -82, -1, -6, 39, 78, -2, 31, 37, 59, -29, 39, 63, 61, -45, 3, -53, -91, -78, -4, 53, 95, -97, -53, 38, 53, 0, 73, 60, -17, -96, 99, 72, -81, -2, -85, -96, -98, 49, 29, -87, 1, 16, 89, -52, -32, 18, -24, 16, -56], [-33, -78, 6, 8, -23, 54, -59, -24, -81, -93, 17, -49, -28, -19, -93, 74, -43, -93, -37, -98, -17, 7, -19, -50, -69, 43, -92, 70, 15, -81, -3, 5, -27, -32, -20, 18, -66, -48, -13, 27, 7, 11, -34, 10, 74, 26, 46, 56, 30, 21, 17, -39, -16, 86, -85, 94, -76, -12, -16, -63, -43, -3, 95, 13, 88, 9, -34, -34, -22, -83, 26, 55, -10, -96, 95, 14, 73, 0, -89, 56, -15, 54, -30, -67, 2, 8, 27, -47, -36, 76, 73, -96, 94, -21, -53, -86, -29, -15, -22, 21], [-78, 32, -40, 98, -7, -79, 14, 62, 28, -57, 97, -2, 61, 2, 44, -59, 93, 72, -85, 63, -78, 3, 97, 7, 62, 7, 68, 82, -86, -62, 68, 34, 23, 27, 33, -22, -2, 13, 72, -85, -65, 57, 52, -50, -100, -76, 44, -6, 99, 32, -20, 88, 70, -70, 25, 38, 11, -36, 32, -68, -98, 72, -18, 63, 99, -60, -91, 64, 57, -39, -30, -27, 30, 12, -66, 95, 60, -89, 0, 35, 10, 84, 62, 50, 74, -23, -57, 57, -70, 70, 76, 83, -80, -30, 1, -45, 89, -76, 58, -55], [56, -95, -4, 29, -88, -4, -92, 15, 39, -81, -30, 31, -32, -67, -81, 86, 61, 57, -51, 44, 17, 89, -9, 80, -78, 11, -26, 25, 19, -19, -8, 96, 48, 35, -94, -37, -1, 36, 24, 77, -19, -72, 55, 86, -35, 62, 95, 2, -55, -64, -18, 48, 55, 93, -47, 53, -28, 72, -1, 59, -15, -2, -21, 77, 3, 24, 40, 51, -65, -41, -25, 87, -74, 88, 4, 14, -17, 56, 35, 0, -51, -40, 65, -81, 69, -40, -47, 27, 13, -33, 25, 23, -70, -47, 69, 38, -11, -18, -36, -56], [45, 83, 80, -31, 89, 92, 19, 86, -33, 31, 83, -9, 57, -61, 84, -18, -52, -8, -90, 33, 59, -46, -56, -75, -2, -61, 99, 30, 53, 78, -29, -29, 76, -29, -27, -43, 46, 35, -25, -33, -34, -80, -49, -68, -84, -64, -25, 52, -1, -53, -70, -60, -70, -95, -63, 50, -6, -50, 77, -72, -100, -89, 16, 41, -86, -65, -35, -19, 89, 0, -99, -90, -96, 67, 65, 4, -96, -15, 10, -51, 0, -4, -25, 46, -40, -15, 94, -54, 84, -99, -67, -87, -32, 87, -26, 64, -33, 57, 52, -45], [63, 13, 26, 97, -66, 56, 8, -37, -3, 98, -68, -64, 89, -87, 21, 30, 52, -8, 0, -62, 99, -10, -8, 71, 90, -71, -1, -52, -43, 11, 13, 31, -90, 92, -68, -75, 45, -33, 41, 53, 16, 66, 14, -37, 5, -75, 89, -76, -34, -56, 57, 4, -15, 69, -64, -2, 76, -52, -45, -75, -18, -98, -76, -58, 48, -59, 55, -97, -65, -83, 22, -99, -53, 71, 92, -70, 99, 54, 84, -40, -4, 0, -41, 59, 73, -77, -57, -34, -96, -27, 15, 97, 92, -29, -70, 91, 0, 9, 46, 11], [42, 31, -96, -23, -97, 30, 76, -74, 58, -68, 71, -29, -37, 9, 26, 87, -12, -45, -46, 21, -14, 60, 93, -85, -47, 65, -62, -31, -9, -71, -87, 63, 53, 1, 41, -19, -38, 88, 78, 15, 45, 34, 15, -56, 98, -9, 45, -86, 66, 20, 29, -93, -18, 29, 44, -86, 94, -21, 4, -60, -82, -38, -94, -49, 81, 36, 8, -34, 81, -100, 54, -1, -51, -20, 1, -77, 72, -30, 62, 65, -25, -41, 0, 93, -55, -70, 29, 88, -33, 95, -68, 33, -82, -15, 44, 33, 86, -36, 85, -31], [-94, 4, -14, -47, -96, -12, -13, -85, 70, 72, -34, 37, -69, 62, -42, 63, -24, -21, -32, -2, -61, -87, 2, 94, 54, -38, -55, 18, -72, -74, 84, -54, 6, 99, -17, -3, -94, -21, 10, -28, 73, -87, 35, 29, -25, -12, -42, -59, 26, -78, -62, 1, -78, -35, 80, 57, -7, -66, -10, -2, 85, 97, -100, -81, 95, 23, 79, 75, -2, -2, 13, 48, 3, -91, 18, 75, -81, -67, 50, -81, 46, 59, 93, 0, 73, -89, -70, 73, -98, 52, -61, -63, 40, -24, 61, -93, -69, 89, -88, -87], [-29, 24, -35, 65, 53, 17, -81, -25, 50, -50, 23, -39, -49, 80, 12, 16, 76, 38, -33, 10, -19, -9, 84, -4, 91, 80, -50, -52, -44, 5, 58, -6, 40, 4, -3, 69, -92, 66, -64, 82, 63, 93, -78, 73, -100, -79, -47, 59, 73, 75, -12, 21, 85, 96, 38, 5, -98, 97, -35, -11, -68, -87, 79, -63, 66, 87, 32, 2, 45, -29, -48, -78, 9, -65, 39, -32, -2, 2, 74, 69, -40, 73, -55, 73, 0, -25, -10, 64, 64, 48, 6, 57, -18, 45, -97, -52, 69, -30, 37, 76], [60, 63, 35, 76, -79, 20, -16, 20, 55, 44, -57, -88, -58, 55, -98, 77, 75, 39, -91, 69, 4, -67, 14, 19, 62, -59, -89, 78, 14, -55, 38, -9, 17, 30, 65, -4, 17, -95, -8, -94, -95, 26, 92, 98, -77, -10, -84, 64, -42, -90, -19, 98, 65, -100, 94, -30, 91, -68, -27, -28, 2, -5, 1, 43, 5, -82, 9, -23, 88, 71, 13, -14, -57, 24, -78, 53, -85, 8, -23, -40, -15, -77, -70, -89, -25, 0, -64, -26, 44, -16, 67, -63, 80, 90, 10, -50, -59, -33, -93, 0], [-96, -73, -48, 85, -11, 93, -86, -77, -3, 40, -23, 97, 86, -29, 7, -93, -38, -37, 7, 57, -19, -79, -51, 80, -64, -38, 23, 78, 75, -1, 72, -79, -27, 52, -24, 19, 83, 92, -79, 33, 14, -30, 18, -12, 60, 60, -57, 50, 88, 79, 81, -24, -31, 69, -54, -46, 21, -50, -100, -47, 66, 0, 88, -50, -42, 70, 17, 0, -62, 89, 36, 86, -26, -32, -67, 28, -96, 27, -57, -47, 94, -57, 29, -70, -10, -64, 0, -60, 16, 37, 14, 67, -99, -88, -12, 20, -49, -85, 94, -13], [63, 30, 73, 47, 82, 99, 94, -61, 35, 23, 0, 39, 48, 71, 51, -4, -35, -48, 47, -73, 98, -49, -45, -69, -44, 2, -7, 39, 68, 48, -100, 81, 6, 12, -97, -62, -71, 10, 94, -6, 63, 64, -15, -91, -81, -4, 4, -41, 61, -73, 91, 89, -16, -88, -22, 12, -54, 69, -84, 11, -91, 8, -27, 28, 34, 46, 7, -37, 8, -5, 49, -93, -53, 24, -15, 88, -98, -47, 57, 27, -54, -34, 88, 73, 64, -26, -60, 0, -98, -81, -79, -44, -78, 11, -85, -11, 69, 28, 91, 47], [15, -92, -36, -32, -78, -47, 21, -98, 45, -22, -91, -39, -96, 49, -7, -43, 91, 23, 81, 71, 90, 27, 5, 44, -37, -63, 19, -99, -20, 39, -68, 38, 54, 94, 0, -94, -42, -61, -27, 4, 37, 28, 88, 22, 1, 45, 88, 68, -84, -81, 15, 34, 92, -90, 33, 87, 66, -7, -63, 4, -55, 88, 19, 82, 84, -41, -5, 34, -4, 47, 42, 54, -77, -77, -13, 32, 49, -36, -70, 13, 84, -96, -33, -98, 64, 44, 16, -98, 0, 63, -34, 36, -76, 89, -35, 1, -20, 52, 43, -24], [22, -98, -4, 91, -74, -91, 35, -100, 69, 94, -52, 64, 60, 0, 66, 93, 44, 16, -27, 10, -61, 36, 60, -77, 79, 93, -80, 4, 71, -15, -1, 77, -56, 11, 52, 80, -67, -47, -1, 7, -86, 40, 80, 64, 83, 99, 64, 33, -20, 44, 44, -54, 17, 96, 72, -69, 40, -37, 47, 74, 73, -92, -57, -55, -96, 79, -45, -25, -72, 65, 68, -49, -72, -55, -97, 22, 29, 76, 70, -33, -99, -27, 95, 52, 48, -16, 37, -81, 63, 0, 43, 68, -54, 56, 74, 46, 24, -62, -31, 94], [-65, 99, -3, -85, 45, -70, 12, 14, 56, 5, -7, -95, -100, 48, 77, -79, -45, -34, 0, 87, 81, -69, -16, -51, -30, -6, 75, 7, -61, 42, 78, 55, -67, -72, 30, -4, -63, 83, 21, -57, 31, 6, 92, -35, -44, 75, 71, 93, 87, -23, 99, -44, -15, 87, 53, -20, 80, 13, 83, -42, -3, 4, 57, -98, 82, -42, 48, -58, -12, -15, 85, 22, 12, -76, 16, -33, -87, 73, 76, 25, -67, 15, -68, -61, 6, 67, 14, -79, -34, 43, 0, -57, 79, -31, -60, -87, 98, 20, 94, 54], [-62, -8, 50, -34, 98, -40, -13, 71, -36, -47, -3, -49, 28, 64, 87, -42, 38, -71, -38, 17, -63, 12, -23, 61, 17, -2, -27, 81, -64, 38, -55, 9, -50, 99, 81, -24, -29, -78, -41, 45, -93, -42, -51, 95, -98, -37, 80, 77, -52, 69, -11, 66, -30, 35, 59, -48, 27, 63, -82, 61, -89, 21, 27, -94, 20, 83, -35, 66, 96, 11, -62, 36, 38, 35, -51, 72, 1, -96, 83, 23, -87, 97, 33, -63, 57, -63, 67, -44, 36, 68, -57, 0, -11, 96, -42, 85, 96, -68, -21, 72], [18, -34, -9, -62, 95, 48, 61, 23, -22, 21, 58, -53, -48, -39, 79, 38, -36, -68, -87, 21, -43, -17, 41, -7, -51, -32, 22, -54, 60, 0, -99, -10, -70, -99, -19, 78, 24, -94, -9, 80, 29, 97, -3, 26, -77, 97, 26, -93, 60, -49, 28, 61, 91, -40, -100, -48, -37, -6, -86, 28, -1, 9, 31, 55, -82, -64, -39, -49, -69, -43, 7, -5, 38, -60, 98, 26, 16, 94, -80, -70, -32, 92, -82, 40, -18, 80, -99, -78, -76, -54, 79, -11, 0, -19, 55, 71, 56, -56, -78, 66], [-58, -75, 17, -36, 4, -69, -86, 94, 54, -69, -41, 95, 72, 23, 31, 0, 51, -58, -41, -12, 69, -39, 38, -64, -83, 60, -36, -11, -1, 14, -73, 27, 49, -14, -82, -37, -86, -51, -40, 54, 15, -9, 32, 54, -69, 27, -24, -61, 1, -53, 59, -93, -55, -14, -27, 13, 92, 10, 1, -83, 6, 25, -82, 22, 54, 8, 7, 57, -94, -96, 29, 47, 77, -27, -81, -25, 89, -21, -30, -47, 87, -29, -15, -24, 45, 90, -88, 11, 89, 56, -31, 96, -19, 0, 74, -63, 97, 5, 2, 60], [66, 24, -24, -1, 53, -3, 19, 35, -8, -24, -47, -15, -56, 17, 20, 55, 61, 99, 36, -79, -7, -36, 87, 19, -13, -51, -18, 23, -75, -1, 39, 99, -13, 23, -32, -90, -3, -12, -78, 55, 8, 70, -7, 12, 34, 28, 8, 58, -23, -13, 56, 39, -79, 4, 65, 57, 95, -73, -13, -87, -19, 46, -19, -79, -99, 70, -46, 27, -33, -54, -80, -3, -59, -43, 8, -2, -52, -53, 1, 69, -26, -70, 44, 61, -97, 10, -12, -85, -35, 74, -60, -42, 55, 74, 0, 7, -9, 56, 59, -86], [-7, -12, 68, -32, 21, -71, 36, -88, -46, -44, 72, -99, 46, -8, 89, 6, -19, -78, -42, 31, -30, -18, -40, 36, -31, 57, 37, -60, -54, -29, -75, 35, 9, 37, -46, -45, 18, 30, 65, -39, -94, 19, 38, -61, -35, 6, -77, -72, 58, 54, 38, 88, -7, -21, -90, -26, 65, 84, -37, -94, 40, -49, -27, 63, -63, 55, 34, 43, -49, -12, -87, 28, 83, 17, 7, 41, -32, -86, -45, 38, 64, 91, 33, -93, -52, -50, 20, -11, 1, 46, -87, 85, 71, -63, 7, 0, 84, 14, -11, -2], [64, 22, 65, -74, -75, 36, -63, -26, 82, 23, -94, -76, -62, -38, 12, -70, 17, 31, 16, 86, -33, -78, 61, 62, 46, 47, 78, -92, -25, -57, 51, 41, 73, -62, -66, 89, -83, 86, 95, 9, -50, -81, -50, -90, -20, -4, 88, -17, -36, 70, -48, -56, 95, 80, 50, -51, 60, 80, 91, 43, 85, -64, -79, -27, 40, -49, 14, 12, -89, -2, -27, -49, 28, -84, -59, -26, 18, -29, 89, -11, -33, 0, 86, -69, 69, -59, -49, 69, -20, 24, 98, 96, 56, 97, -9, 84, 0, 84, 53, -70], [-67, 48, -46, -71, 0, -52, -74, 71, 50, -41, 67, -37, 6, 37, 27, 19, 8, -56, -94, -95, 96, -93, -78, -30, -87, -81, -79, -71, -78, 46, -26, 92, 10, 35, 87, 93, 62, 8, 9, -63, -76, 18, -98, 43, -58, 94, 83, -59, -6, 62, -80, 72, -61, 98, -24, -25, 71, 0, -80, 13, 53, -87, 71, 94, 66, -79, 36, -72, -12, -17, -21, 46, 61, 99, -38, 78, -24, -15, -76, -18, 57, 9, -36, 89, -30, -33, -85, 28, 52, -62, 20, -68, -56, 5, 56, 14, 84, 0, -97, -80], [30, -55, 72, 63, -78, 37, -32, -79, 34, 20, -95, 82, -32, -74, -65, -3, 95, 97, 89, 87, -58, -97, 86, -31, 68, -44, -32, -49, -3, -37, 98, -72, -21, 90, 84, 32, 55, 81, -71, -1, -90, -70, -62, -28, 83, -10, -58, 77, -5, -45, -40, -41, -6, -80, 7, 14, 1, 57, 91, 63, 69, -18, -60, -67, -40, 16, 25, 71, -77, -60, 88, 35, -52, 36, 1, -97, 16, -22, 58, -36, 52, 46, 85, -88, 37, -93, 94, 91, 43, -31, 94, -21, -78, 2, 59, -11, 53, -97, 0, -38], [-100, 88, 29, 37, 32, -78, -2, -12, 56, -33, 8, -57, -92, 60, 45, 83, 32, -73, -58, -40, 60, -25, 91, 10, 72, -36, -72, -30, -19, -19, -37, 54, 77, 69, 1, -74, -49, -36, 95, 36, 54, -72, -48, 27, 31, 64, 49, -15, -74, -14, -40, 30, 93, -89, 89, -73, 63, -37, -36, 29, -18, 11, 82, 68, 46, 32, -5, -74, 10, 9, -52, -60, -20, -12, 68, 45, -56, 21, -55, -56, -45, 11, -31, -87, 76, 0, -13, 47, -24, 94, 54, 72, 66, 60, -86, -2, -70, -80, -38, 0]]

    if algorithm == "Genetic Algorithm version1":
        start_time = time.time()
        res = genetic_algorithm_1(500, 100, matrix, seats_per_table)
        end_time = time.time()
        print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
        return f"A executar Genetic Algorithm version1 com {num_guests} convidados e {seats_per_table} assentos por mesa. Resultado: {res}"
    elif algorithm == "Genetic Algorithm version2":
        start_time = time.time()
        res = genetic_algorithm_2(5000, 100, matrix, seats_per_table)
        end_time = time.time()
        print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
        return f"A executar  Genetic Algorithm version2 com {num_guests} convidados e {seats_per_table} assentos por mesa. Resultado: {res}"
    elif algorithm == "Simulated Annealing":
        start_time = time.time()
        res, score = simmulated_annealing(matrix_, seats_per_table)
        end_time = time.time()
        print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
        return f"A executar Simulated Annealing com {num_guests} convidados e {seats_per_table} assentos por mesa. Score: {score}. Resultado: {res}"
    elif algorithm == "Tabu Search":
        start_time = time.time()
        res = tabu_search(matrix_, seats_per_table, max_iterations=1000, tabu_tenure=7, max_no_improve=100)
        end_time = time.time()
        print(f"Tempo de execução: {end_time - start_time:.6f} segundos")
        return f"A executar Tabu Search com {num_guests} convidados e {seats_per_table} assentos por mesa. Resultado: {res}"
    else:
        return "Erro: Algoritmo desconhecido!"
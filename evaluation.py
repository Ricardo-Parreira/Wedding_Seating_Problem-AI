test_arragenment = [[0,2], [1,3]]

test_matrix = [[0,-10, 100, -100],
              [10, 0, 0, 100],
              [0,0,0,0],
              [-100, 100, -50, 0]]

def evaluate_table(tables, matrix):
    score = 0

    #nao me parece que isto seja a forma mais eficaz mas é só para termos uma base
    for table in tables:
        for guest in table:
            for neighbor in table:
                if guest != neighbor:
                    score += test_matrix[guest][neighbor]

    average = score / len(tables)
    return average


ave = evaluate_table(test_arragenment, test_matrix)
print(ave)
    
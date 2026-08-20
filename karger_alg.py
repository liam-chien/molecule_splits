import numpy as np

# Perform Karger's algorithm
# WARNING: This will clear the adjacency matrix, so make sure you pass it a COPY of anything you care about
def generate_karger(adj_mat, test_p, seed):

    gen = np.random.default_rng(seed)
    
    # Number of molecules
    num_nodes = len(adj_mat)

    col_sums = np.sum(adj_mat, axis=0)
    free_mols = np.where(col_sums == 0)[0]

    # Stop the algorithm once a supernode reaches this size
    SUPERNODE_CUTOFF = test_p * num_nodes - len(free_mols)
    
    # Remove the supernode cutoff for traditional karger's alg
    # SUPERNODE_CUTOFF = num_nodes + 1 
    
    # Each node starts off on its own (N supernodes), and they will merge until one supernode reaches SUPERNODE_CUTOFF size
    supernodes = [{i} for i in range(0, num_nodes)]

    # While there are supernodes to be merged
    while len(supernodes) > 2:
    
        p, q = select_rand_edge(adj_mat, gen)

        # Opportunity for multi-threading:
        merge_nodes(p, q, adj_mat)
        new_supernode = merge_sets(p, q, supernodes)

        # If a supernode exceeds the cutoff size
        if len(new_supernode) >= SUPERNODE_CUTOFF:
            new_supernode.update(free_mols)
            return list({i for i in range(0, num_nodes)} - new_supernode), list(new_supernode)
        

    return list(supernodes[0]).extend(free_mols), list(supernodes[1])


# Picks a random weighted edge
def select_rand_edge(adj_mat, gen):
    
    # Find the probablity of the selction being in a certain column and pick a column accordingly
    col_sums = np.sum(adj_mat, axis=0)
    mat_total = np.sum(col_sums)

    assert mat_total != 0, "The graph has no edges"
    
    col_idx = gen.choice(len(adj_mat), p=(col_sums / mat_total))

    # From the selected column choose a weighted random value
    col = adj_mat[:, col_idx]
    row_idx = gen.choice(len(adj_mat), p=(col / np.sum(col)))

    return row_idx, col_idx


# Merges nodes i and j into one supernode in the adjacency matrix, adding the weights of merged edges
# Adjacency matrix must be upper triangular
# The supernode becomes represented by i, and j loses all connections
def merge_nodes(i, j, upper_mat):

    num_nodes = len(upper_mat)

    # Break the edge between i and j
    upper_mat[i, j] = 0
    upper_mat[j, i] = 0

    for k in range(num_nodes):

        
        # Merge the edges, increasing the weight of all i-edges in accordance with the correspoinding j-edge
        upper_mat[i, k] += upper_mat[j, k]
        upper_mat[k, i] += upper_mat[k, j]

        # Remove all j-edges, leaving j floating sad and alone
        upper_mat[j, k] = 0
        upper_mat[k, j] = 0




# Merges the set containing val1 with the set containing val2
# Will update set_list to contain the new set and remove the old sets
def merge_sets(val1, val2, set_list):


    set1 = None
    set2 = None
    
    for mol_set in set_list:
        
        if val1 in mol_set:
            # Found the set of value 1
            set1 = mol_set
            
            if set2 != None:
                break
        elif val2 in mol_set:
            # Found the set of value 2
            set2 = mol_set

            if set1 != None:
                break

    # Add all of set2 to set1
    set1.update(set2)

    # Remove set2
    set_list.remove(set2)

    return set1

    

    
            
#!/usr/bin/env python
import sys
from Bio.PDB import PDBParser
import numpy
import MultiNEAT as NEAT
import math
import pycuda.driver
import pycuda.autoinit
import pycuda.gpuarray
from pycuda.compiler import SourceModule
from pycuda.curandom import XORWOWRandomNumberGenerator
import pycuda.cumath
import pycuda.elementwise

sys.path.append('/usr/local/cuda/bin')
sys.path.append('/usr/local/cuda/lib64')

# NEAT activation kernel
gpu_sim_mod = SourceModule("""
__global__ void gpu_sim(float* d_inputs,
                float* d_results, float* d_m_source, float* d_m_weight,
                float* d_m_target,
                float* d_m_a, float* d_m_b, int c_size,
                int n_size, int i_size, int num_inputs, float* signal_malloc,
                float* activesum_malloc, int a_size){

    int tx(blockIdx.x * blockDim.x + threadIdx.x);

    if (tx < a_size){
        float d = d_inputs[3*tx + 1];
        float y = 2.0;
        if (d < 15.0){

            float* m_signal(signal_malloc + tx * c_size);

            float* m_activesum(activesum_malloc + tx * c_size);

            for (int i = 0; i < 3; i++){
                for (int i = 0; i < c_size; i++){
                    int id = d_m_source[i];
                    m_signal[i] = (d_inputs[3*tx + id]) * d_m_weight[i];
                }

                for (int i = 0; i < c_size; i++){
                    int id = d_m_target[i];
                    m_activesum[id] = m_activesum[id] + m_signal[i];
                }

                for (unsigned int i = num_inputs; i < n_size; i++)
                {
                    float x = m_activesum[i];
                    x = x/1000.0;
                    float a = d_m_a[i];
                    float b = d_m_a[i];
                    y = exp(-a*x*x - b);
                    //y = x;
                }
            }

            d_results[tx] = y;

        } else{

            d_results[tx] = 0.0;

        }
    }
}
""")

gpu_sim = gpu_sim_mod.get_function("gpu_sim")

# Kernel for updating distances from probabilities
update_distance_mod = SourceModule("""

__global__ void update_distance(int i_size, float* d_results_copy,
float* d_results, float* rand_array, int a_size){

    int tx(blockIdx.x * blockDim.x + threadIdx.x);
    if (tx < a_size){
        float d = 0.0;
        float p = d_results[tx];
        //float rand = curand_uniform(&s);
        float rand = rand_array[tx];
        if (p > rand){
            d = -0.5;
        } else if (p != 0.0){
            d = 0.5;
        }

        if (!((d_results_copy[tx] <= 0.5) && (d == -0.5)))
            d_results_copy[tx] = d_results_copy[tx] + d;

    }

}

""")

update_distance = update_distance_mod.get_function("update_distance")


def name_to_int(atom_type, neighbor_type):
    """Converts atom pair to integer.
    :param atom_type: :type: string: Name of first atom.
    :param neighbor_type: :type: string: Name of second atom.
    :return: :type: integer: Integer representing atom pair.
    """

    atom_pair = 0
    if atom_type == "C" and neighbor_type == "C":
        atom_pair = 1
    elif (atom_type == "C" or neighbor_type == "C") and (atom_type == "N" or neighbor_type == "N"):
        atom_pair = 2
    elif (atom_type == "C" or neighbor_type == "C") and (atom_type == "O" or neighbor_type == "O"):
        atom_pair = 3
    elif (atom_type == "C" or neighbor_type == "C") and (atom_type == "H" or neighbor_type == "H"):
        atom_pair = 4
    elif (atom_type == "C" or neighbor_type == "C") and (atom_type == "S" or neighbor_type == "S"):
        atom_pair = 5
    elif atom_type == "N" and neighbor_type == "N":
        atom_pair = 6
    elif (atom_type == "N" or neighbor_type == "N") and (atom_type == "O" or neighbor_type == "O"):
        atom_pair = 7
    elif (atom_type == "N" or neighbor_type == "N") and (atom_type == "H" or neighbor_type == "H"):
        atom_pair = 8
    elif (atom_type == "N" or neighbor_type == "N") and (atom_type == "S" or neighbor_type == "S"):
        atom_pair = 9
    elif atom_type == "O" and neighbor_type == "O":
        atom_pair = 10
    elif (atom_type == "O" or neighbor_type == "O") and (atom_type == "H" or neighbor_type == "H"):
        atom_pair = 11
    elif (atom_type == "O" or neighbor_type == "O") and (atom_type == "S" or neighbor_type == "S"):
        atom_pair = 12
    elif atom_type == "H" and neighbor_type == "H":
        atom_pair = 13
    elif (atom_type == "H" or neighbor_type == "H") and (atom_type == "S" or neighbor_type == "S"):
        atom_pair = 14
    elif atom_type == "S" and neighbor_type == "S":
        atom_pair = 15

    return atom_pair


def rmsd_calc(packed_input, packed_ref, num_atoms):
    """Calculate RMSD between distance matrices of target and simulated proteins.
    :param packed_input: :type ndarray: One dimensional reduced array of distance matrix of simulated protein.
    :param packed_ref: :type ndarray: " of target protein.
    :param num_atoms: :type int: Number of atoms in simulation.
    :return: :type float: RMSD between proteins.
    """

    diff_sq = packed_input - packed_ref

    diff_sq = diff_sq**2

    diff_sq_sum = (numpy.sum(diff_sq) / num_atoms) / 2

    rmsd = math.sqrt(diff_sq_sum)

    return rmsd


def ind_of(i, j):
    """Convert index of normal matrix to reduced array index. i must be less than j.
    """

    return i+j(j-1)/2


def sim_loop(packed_input, packed_ref, packed_np, num_atoms, net):
    """Main simulation loop.
    :param packed_input: :type ndarray: Reduced array of distance matrix for simulation starting point.
    :param packed_ref: :type ndarray: Reduced array of distance matrix for target structure.
    :param packed_np: :type ndarray: Reduced array of integer mapping of atom pairs.
    :param num_atoms: :type ndarray: Number of atoms in the simulation.
    :param net: :type NEAT.NeuralNetwork: NEAT neural network to use for state transitions.
    :return: RMSD between simulation endpoint and target structure.
    """

    # Allocate arrays for input and output
    packed_input = packed_input.astype(numpy.float32)
    full_input = numpy.zeros(packed_input.size*3, dtype=numpy.float32)
    updating = numpy.zeros(packed_input.size, dtype=numpy.float32)
    probabilities = numpy.zeros(packed_input.size, dtype=numpy.float32)
    d_updating = pycuda.gpuarray.to_gpu(updating)
    updated = numpy.zeros(packed_input.size, dtype=numpy.float32)
    d_updated = pycuda.gpuarray.to_gpu(packed_input)
    updating_host = numpy.zeros(packed_input.size, dtype=numpy.float32)
    i_size = full_input.size
    a_size = i_size/3

    # Convert to full network input
    i = 0
    j = 0
    for distance in packed_input:
            pair_id = packed_np[j]
            full_input[i] = pair_id
            full_input[i + 1] = distance
            full_input[i + 2] = 1
            i += 3
            j += 1

    d_full_input = pycuda.gpuarray.to_gpu(full_input)

    # Get parameters of network
    net.ActivateFast()
    results = net.Output()

    # Make various constants
    num_inputs_float = results[len(results) - 1]
    n_size_float = results[len(results) - 2]
    c_size_float = results[len(results) - 3]
    c_size_int = int(c_size_float)
    n_size_int = int(n_size_float)
    num_inputs = numpy.int32(num_inputs_float)
    n_size = numpy.int32(n_size_float)
    c_size = numpy.int32(c_size_float)

    # Allocate arrays for network parameters
    d_m_weight = numpy.zeros(c_size, dtype=numpy.float32)
    d_m_source = numpy.zeros(c_size, dtype=numpy.float32)
    d_m_target = numpy.zeros(c_size, dtype=numpy.float32)
    d_m_a = numpy.zeros(n_size, dtype=numpy.float32)
    d_m_b = numpy.zeros(n_size, dtype=numpy.float32)

    # Populate those arrays
    for i in range(len(results)):
        if i < c_size:
            d_m_weight[i] = results[i]
        elif i < c_size*2:
            d_m_source[i - c_size_int] = results[i]
        elif i < c_size*3:
            d_m_target[i - c_size_int*2] = results[i]
        elif i < (c_size*3 + n_size):
            d_m_a[i - c_size_int*3] = results[i]
        elif i < (c_size*3 + 2*n_size - 3):
            d_m_b[i - (c_size_int*3 + n_size_int)] = results[i]

    # Grid and thread size
    grid = int(math.ceil(i_size/1024))
    threads_per_launch_malloc = int((grid * 1024)*c_size)
    grid_s = int(math.ceil(i_size/512))
    threads_per_launch_malloc_s = int((grid_s * 512)*c_size)

    print "\n C size:"
    print str(c_size)

    print "\n N size:"
    print str(n_size)

    # Bring network parameters to device
    dev_m_weight = pycuda.gpuarray.to_gpu(d_m_weight)
    dev_m_source = pycuda.gpuarray.to_gpu(d_m_source)
    dev_m_target = pycuda.gpuarray.to_gpu(d_m_target)
    dev_m_a = pycuda.gpuarray.to_gpu(d_m_a)
    dev_m_b = pycuda.gpuarray.to_gpu(d_m_b)
    signal_malloc = numpy.zeros(threads_per_launch_malloc_s)
    activesum_malloc = numpy.zeros(threads_per_launch_malloc_s)
    signal_malloc = signal_malloc.astype(numpy.float32)
    activesum_malloc = activesum_malloc.astype(numpy.float32)
    d_signal_malloc = pycuda.gpuarray.to_gpu(signal_malloc)
    d_activesum_malloc = pycuda.gpuarray.to_gpu(activesum_malloc)

    rand = pycuda.curandom.XORWOWRandomNumberGenerator(seed_getter=None, offset=0)

    rand_array = rand.gen_uniform(a_size, dtype=numpy.float32)

    total_iter = 0
    n_steps = 10000
    i_size = numpy.int32(i_size)
    a_size = numpy.int32(a_size)

    # For all your diagnostic needs
    # print str(d_full_input)
    # print str(d_updating)
    # print str(dev_m_source)
    # print str(dev_m_weight)
    # print str(dev_m_target)
    # print str(dev_m_a)
    # print str(dev_m_b)
    # print str(c_size)
    # print str(n_size)
    # print str(i_size)
    # print str(num_inputs)
    # print str(d_signal_malloc)
    # print str(d_activesum_malloc)
    # print str(a_size)

    # Run simulation for n_steps steps
    while total_iter <= n_steps:

        gpu_sim(d_full_input, d_updating, dev_m_source, dev_m_weight,
                dev_m_target, dev_m_a, dev_m_b, c_size,
                n_size, i_size, num_inputs, d_signal_malloc, d_activesum_malloc, a_size, block=(512, 1, 1),
                grid=(grid_s, 1))

        pycuda.driver.Context.synchronize()

        update_distance(i_size, d_updated, d_updating, rand_array, a_size, block=(512, 1, 1), grid=(grid_s, 1))

        pycuda.driver.Context.synchronize()

        total_iter += 1

    # Get calculated probabilites from device
    d_updating.get(probabilities)

    # Get new atom positions from device
    d_updated.get(updated)

    # Uncomment to print all distances.
    # for d in updated:
        # print str(d)

    # Uncomment to print probability values.
    # for d in probabilities:
    #     if d != 0.0:
    #         print str(d)

    return rmsd_calc(updated, packed_ref, num_atoms)


def build_matrices():
    """This builds the reduced arrays representing the distance matrices of simulation starting point and target structure.
    :return: A list containing arrays for [starting structure, reference structure, atom pairs, num atoms].
    """

    sys.path.append('/home/dan/pf/NoveltySearch')

    sys.path.append('/home/dan/pf/PeptideBuilder/PeptideBuilder')

    pdb = '4PWQH.pdb'

    pdb_build = 'sequence_4PWQH.pdb'

    parser = PDBParser()

    print "Building reference structure... \n"

    structure_ref = parser.get_structure('REF', pdb)

    print "Building simulation structure... \n"

    structure = parser.get_structure('STRCT', pdb_build)

    num_atoms = 0
    i_iter = 0
    i_iter_ref = 0
    iter_init = 0

    print "Initializing distance matrix and arrays... \n"

    for atom in structure.get_atoms():
        num_atoms += 1

    # Actually a multidimensional array
    input_matrix = numpy.zeros((num_atoms, num_atoms))
    ref_matrix = numpy.zeros((num_atoms, num_atoms))
    sn_array = numpy.zeros(num_atoms)
    name_array = numpy.empty(num_atoms, numpy.dtype(str))
    name_pair_matrix = numpy.zeros((num_atoms, num_atoms))
    packed_input = numpy.zeros(num_atoms+num_atoms*(num_atoms-1)/2)
    packed_ref = numpy.zeros(num_atoms+num_atoms*(num_atoms-1)/2)
    packed_np = numpy.zeros(num_atoms+num_atoms*(num_atoms-1)/2)

    for atom in structure.get_atoms():
        sn_array[iter_init] = atom.get_serial_number()
        name_array[iter_init] = atom.element
        iter_init += 1

    print "Building distance matrix for simulation structure... \n"

    # Build distance matrix for prediction
    for atom in structure.get_atoms():
        j_iter = 0
        for atom_d in structure.get_atoms():
            distance = atom-atom_d
            input_matrix[i_iter][j_iter] = distance
            j_iter += 1
        i_iter += 1

    pack_iter = 1
    ind = 0

    print "Reducing... \n"

    for i in range(num_atoms):
        for j in range(pack_iter):
            packed_input[ind] = input_matrix[j][i]
            ind += 1
        pack_iter += 1

    print "Building distance matrix for reference structure... \n"

    # Build distance matrix for reference
    for atom in structure_ref[0]['A'].get_atoms():
        j_iter_ref = 0
        for atom_d in structure_ref[0]['A'].get_atoms():
            distance = atom-atom_d
            ref_matrix[i_iter_ref][j_iter_ref] = distance
            j_iter_ref += 1
        i_iter_ref += 1
    cq_iter_1 = 0

    print "Reducing... \n"

    pack_iter_ref = 1
    ind_ref = 0

    for i in range(num_atoms):
        for j in range(pack_iter_ref):
            packed_ref[ind_ref] = ref_matrix[j][i]
            ind_ref += 1
        pack_iter_ref += 1

    for row in input_matrix:
        cq_iter_2 = 0
        an = name_array[cq_iter_1]
        for distance in row:
            nn = name_array[cq_iter_2]
            name_pair_matrix[cq_iter_1][cq_iter_2] = name_to_int(an, nn)
            cq_iter_2 += 1
        cq_iter_1 += 1

    pack_iter_n = 1
    ind_n = 0

    for i in range(num_atoms):
        for j in range(pack_iter_n):
            packed_np[ind_n] = name_pair_matrix[j][i]
            ind_n += 1
        pack_iter_n += 1

    return_arrays = [packed_input, packed_ref, packed_np, num_atoms]

    return return_arrays


def run_sim(net, return_arrays):
    """Executes simulation and returns the fitness of endpoint.
    :param net: :type NEAT.NeuralNetwork: NEAT neural network to use for simulation.
    :param return_arrays: :type list: Arrays generated by build_matrices().
    :return: :type float: Fitness of simulation endpoint.
    """

    print "\n Entering simulation... \n"

    packed_input = return_arrays[0]
    packed_ref = return_arrays[1]
    packed_np = return_arrays[2]
    num_atoms = return_arrays[3]

    rmsd = sim_loop(packed_input, packed_ref, packed_np, num_atoms, net)

    print "\n RMSD:"
    print str(rmsd)

    fitness = 13000 - rmsd

    return fitness


def train():
    """Evolve NEAT NNs.
    """

    params = NEAT.Parameters()

    params.PopulationSize = 100

    print "Creating genome... \n"

    genome = NEAT.Genome(0, 3, 5, 1, False,
                         NEAT.ActivationFunction.UNSIGNED_GAUSS, NEAT.ActivationFunction.UNSIGNED_GAUSS, 1, params)

    print "Initializing population... \n"

    pop = NEAT.Population(genome, params, True, 1.0)

    print "Beginning evolution... \n"

    return_arrays = build_matrices()

    for generation in range(300):

        # Retrieve a list of all genomes in the population
        genome_list = NEAT.GetGenomeList(pop)

        # Apply the evaluation function to all genomes
        for genome in genome_list:
            net = NEAT.NeuralNetwork()
            genome.BuildPhenotype(net)
            fitness = run_sim(net, return_arrays)
            genome.SetFitness(fitness)
            print "\n Generation:"
            print str(generation + 1)

        # Advance to the next generation
        print "Genome evaluated."
        pop.Epoch()


if __name__ == "__main__":

    train()

pycuda.autoinit.context.detach()

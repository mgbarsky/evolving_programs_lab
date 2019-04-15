from random import random,randint,choice
from copy import deepcopy
from math import log


class fwrapper:
    """
    A wrapper for the functions that will be used on function nodes.
    Its member variables are the name of the function,
    the function itself, and the number of parameters it takes.
    """
    def __init__(self,function,childcount,name):
        self.function=function
        self.childcount=childcount
        self.name=name


class node:
    """
    The class for function nodes (nodes with children).
    This is initialized with an fwrapper.
    When evaluate is called, it evaluates the child nodes
    and then applies the function to their results.
    """
    def __init__(self,fw,children):
        self.function=fw.function
        self.name=fw.name
        self.children=children

    def evaluate(self,inp):
        results=[n.evaluate(inp) for n in self.children]
        return self.function(results)

    def display(self,indent=0):
        print ((' '*indent)+self.name)
        for c in self.children:
            c.display(indent+4)
    

class paramnode:
    """
    The class for nodes that only return one of the parameters passed to the program.
    Its evaluate method returns the parameter specified by idx.
    """
    def __init__(self,idx):
        self.idx=idx

    def evaluate(self,inp):
        return inp[self.idx]

    def display(self,indent=0):
        print ('%sp%d' % (' '*indent,self.idx))
    
    
class constnode:
    """
    Node that returns a constant value.
    The evaluate method simply returns the
    value with which it was initialized.
    """
    def __init__(self,v):
        self.v=v

    def evaluate(self,inp):
        return self.v

    def display(self,indent=0):
        print ('%s%d' % (' '*indent,self.v))


def print_expression(tree_node):
    if type(tree_node) is paramnode:
        print('p%d' % tree_node.idx, end="")
        return
    if type(tree_node) is constnode:
        print('%d' % tree_node.v, end="")
        return

    # in-order traversal
    num_children = len(tree_node.children)
    if num_children == 2: # add, mult, subtract
        left_child = tree_node.children[0]
        if left_child:
            print(" (", end="")
            print_expression(left_child)
            print(") ", end="")

        print(tree_node.name, end="")

        right_child = tree_node.children[1]
        if right_child:
            print(" (", end="")
            print_expression(right_child)
            print(") ", end="")
    else: # this is if expression
        print(tree_node.name, end="")
        cond_child = tree_node.children[0]
        if cond_child:
            print(" (", end="")
            print_expression(cond_child)
            print(") ", end="")
        print(" ? ", end="")
        true_child = tree_node.children[1]
        if true_child:
            print(" (", end="")
            print_expression(true_child)
            print(") ", end="")
        print(" : ", end="")
        false_child = tree_node.children[2]
        if false_child:
            print(" (", end="")
            print_expression(false_child)
            print(") ", end="")


def create_functions():
    """
    Creates some sample functions
    :return: tuple of the functions to choose from
    """
    addw=fwrapper(lambda l:l[0]+l[1],2,'add')
    subw=fwrapper(lambda l:l[0]-l[1],2,'subtract')
    mulw=fwrapper(lambda l:l[0]*l[1],2,'multiply')

    ifw=fwrapper(lambda l:  l[1] if l[0]>0 else l[2],3,'if')

    gtw=fwrapper(lambda l: 1 if l[0]>l[1] else 0,2,'isgreater')

    return (addw, mulw, ifw, gtw, subw)


def exampletree():
    (addw, mulw, ifw, gtw, subw) = create_functions()
    return node(ifw,[
                  node(gtw,[paramnode(0),constnode(3)]),
                  node(addw,[paramnode(1),constnode(5)]),
                  node(subw,[paramnode(1),constnode(2)]),
                  ]
              )


def make_random_tree(pc, maxdepth=4, fpr=0.5, ppr=0.6):
    """
    This function creates a node with a random function and then looks to see
    how many child nodes this function requires.
    For every child node required, the function
    calls itself to create a new node.
    :param pc: parameter count - how many parameters function takes
    :param maxdepth: max tree depth
    :param fpr: probability that the new node will be a function node
    :param ppr: probability that the new node will be a parameter node, not a const
    :return: root of the program tree
    """
    flist = create_functions()
    if random() < fpr and maxdepth > 0:
        f = choice(flist)
        children = [make_random_tree(pc,maxdepth-1,fpr,ppr)
                      for i in range(f.childcount)]
        return node(f,children)
    elif random() < ppr:
        return paramnode(randint(0,pc-1))
    else:
        return constnode(randint(0,10))
              

def hidden_function(x,y):
    return x**2+2*y+3*x+5


def build_dataset(n=200):
    rows = []
    for i in range(n):
        x = randint(0, 40)
        y = randint(0, 40)
        rows.append([x, y, hidden_function(x ,y)])
    return rows


def mutate(t, pc, mutation_rate=0.1):
    if random() < mutation_rate:
        return make_random_tree(pc)
    else:
        result = deepcopy(t)
        if hasattr(t, "children"):
            result.children = [mutate(c, pc, mutation_rate) for c in t.children]
        return result


def crossover(t1, t2, probswap=0.7, top=1):
    if random() < probswap and not top:
        return deepcopy(t2)
    else:
        result=deepcopy(t1)
        if hasattr(t1,'children') and hasattr(t2,'children'):
            result.children = [crossover(c, choice(t2.children), probswap, 0)
                                    for c in t1.children]
        return result


def scorefunction(tree,data):
    dif = 0
    for x in data:
        v = tree.evaluate([x[0], x[1]])
        dif += abs(v-x[2])

    return dif


def get_dataset_rankfunction(dataset):
    def rankfunction(population):
        scores=[(scorefunction(t,dataset),t) for t in population]
        scores.sort(key=lambda x: x[0])
        return scores
    return rankfunction
    

def evolve(pc,popsize,rankfunction,maxgen=500,
           mutationrate=0.1,breedingrate=0.4,pexp=0.7,pnew=0.05):
    # Returns a random number, tending towards lower numbers.
    # The lower pexp is, more lower numbers you will get
    def selectindex():
        return int(log(random())/log(pexp))

    # Create a random initial population
    population=[make_random_tree(pc) for i in range(popsize)]

    # main loop
    for i in range(maxgen):
        scores=rankfunction(population)
        print ("generation", i, ", score =", scores[0][0])
        if scores[0][0]==0:
            break
    
        # The two best always make it
        newpop=[scores[0][1],scores[1][1]]
    
        # Build the next generation
        while len(newpop)<popsize:
            if random()>pnew:
                newpop.append(mutate(
                      crossover(scores[selectindex()][1],
                                 scores[selectindex()][1],
                                  probswap=breedingrate),
                                  pc,mutation_rate=mutationrate))
            else:
                # Add a random node to mix things up
                newpop.append(make_random_tree(pc))
        
        population=newpop

    scores[0][1].display()
    return scores[0][1]



def main():
    print("Example of function tree from the lecture")
    lecture_tree = exampletree()
    lecture_tree.display()
    print(lecture_tree.evaluate([2,3]))
    print(lecture_tree.evaluate([5, 3]))

    return
    print()
    print("Making random program with 2 parameters")
    rtree = make_random_tree(2)
    rtree.display()
    print("Evaluating with (2,3)")
    print(rtree.evaluate([2,3]))
    print_expression(rtree)

    print()
    print("Generating dataset and taking a random guess")
    dataset = build_dataset()
    print("My guess gives score",scorefunction(rtree, dataset))

    print()
    print("Trying mutation")
    muttree = mutate(rtree, 2, mutation_rate=1.0)
    print("Mutated gives score", scorefunction(muttree, dataset))

    print()
    print("Evolution in action")
    rf = get_dataset_rankfunction(build_dataset())

    # pexp - the higher - the more elitist are selected
    # pnew - probability of new random program to be introduced
    sol = evolve(2, 500, rf, mutationrate=0.2, breedingrate=0.1, pexp=0.7, pnew=0.1)
    print_expression(sol)


if __name__ == "__main__":
    main()


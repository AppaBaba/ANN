# Simple forward propagation neural network
# three layers input, hidden and output

# sig   Sigmoid function
# slope slope or  derivative of Sigmoid function
# dti   dataset for training input
# dto   dataset for training output
# il    input layer
# hl    hidden layer
# ol    output layer
# sw0   synaptic weight0
# sw1   synaptic weight1
# olerr output layer error
# oldlt output layer delta
# hlerr hidden layer error
# hldlt hidden layer delta

from numpy import exp, array, random, dot, mean, abs

def sig(x):
    return 1/(1 + exp(-x))

def slope(x):
    return x*(1-x)
    
    
dti = array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])               
dto = array([[0],[1],[1],[0]])
random.seed(1)
sw0 = 2*random.random((3,4)) - 1
sw1 = 2*random.random((4,1)) - 1

for i in range(10000):
    il = dti
    hl = sig(dot(il,sw0))
    ol = sig(dot(hl,sw1))
    olerr = dto - ol  
    oldlt = olerr*slope(ol)
    hlerr = oldlt.dot(sw1.T)
    hldlt = hlerr * slope(hl)
    sw1 += hl.T.dot(oldlt)
    sw0 += il.T.dot(hldlt)

print('Output')
print(ol)

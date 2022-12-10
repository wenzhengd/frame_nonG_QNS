from numpy.linalg import matrix_rank
import numpy as np

MATRIX_RANK_TOL = None

#################################################################################################################
##  Modified from https://stackoverflow.com/questions/28816627/how-to-find-linearly-independent-rows-from-a-matrix
#################################################################################################################
def indep_rows(M):
    """
    given the linear indendent rows of a (rectangle) matrix M. 
    """
    non_zero_index=0
    while np.array_equal(M[non_zero_index],np.zeros(np.shape(M)[1])):
        non_zero_index +=1
    LI=[M[non_zero_index]] # Need to take the 1st row of M that is NOT zero-string
    for i in range(non_zero_index,np.shape(M)[0]):
        tmp=[]
        for r in LI:
            tmp.append(r)
        tmp.append(M[i])                #set tmp=LI+[M[i]]
        if matrix_rank(tmp, tol =MATRIX_RANK_TOL)>len(LI):    #test if M[i] is linearly independent from all (row) vectors in LI
            LI.append(M[i])             #note that matrix_rank does not need to take in a square matrix
    return LI                           #return set of linearly independent (row) vectors

#Examples

mat=[[1,2,3,4],[4,5,6,7],[5,7,9,11],[2,4,6,8]]
indep_rows(mat)

mat_p=[[0,0,0,0],[1,2,3,4],[4,5,6,7],[5,7,9,11],[2,4,6,8]]
indep_rows(mat_p)

mat2=[[2,0,3,0],[0,1,1,3],[2,1,4,3],[1,2,3,4]] # This is the example from Mathematica Subspace basis 
###################### https://resources.wolframcloud.com/FunctionRepository/resources/SubspaceBasis/
indep_rows(mat2)

mat3=[[2,0,3,0],[0,1,1,3],[2,1,4,3],[1,2,3,4],[1,2,3,4], [5,6,7,8]]
indep_rows(mat3)
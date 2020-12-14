import copy, math

class Matrix:

    def __init__(self, file=None, array=None):

        self.row = 0
        self.column = 0
        self.matrix = []

        if file:
            self.matrix = self.load(file)
            self.row = len(self.matrix)
            self.column = len(self.matrix[0])

        elif array:
            self.matrix = array
            self.row = len(self.matrix)

            if type(array[0]) is int:
                self.column = 1
            else:
                self.column = len(self.matrix[0])

        else:
            raise ValueError('Illegal number of arguments') 


    def __add__(self, other):
        if ((self.row, self.column) != (other.row, other.column)):
            raise ValueError('Matrices have different dimensions')

        mat = [[self.matrix[i][j] + other.matrix[i][j] for j in range                               (self.column)] for i in range(self.row)]

        return Matrix(array = mat)


    def __mul__(self, other):
        if (type(other) is Matrix):
            result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*other.matrix)] for X_row in self.matrix]
            
            return Matrix(array=result)

        else:
            self.matrix = [[other*num for num in row] for row in self.matrix]

            return self


    def __sub__(self, other):
        other = other * (-1)
        return self + other


    def __eq__(self, other):
        return self.matrix == other.matrix


    def load(self, file):
        matrix = []
        with open(file, 'r') as fp:
            for line in fp:
                matrix.append([float(x) for x in line.split(' ')])

            return matrix


    def clone(self):
        other = copy.deepcopy(self.matrix)
        return Matrix(array=other)


    def print(self):
        for x in self.matrix:
            print(x)
        print()


    def save(self, file):
        with open(file, 'w') as dat:
            for row in self.matrix:
                for num in row:
                    dat.write(str(num) + ' ')
                dat.write('\n')


    def transpone(self):
        mx_transposed = [[self.matrix[i][j] for i in range(self.row)] for j in range(self.column)]
        
        self.matrix = mx_transposed
        (self.row, self. column) = (self.column, self.row)

        return self


    def luDecomposition(self):
        eps = 10**(-15)

        if (self.row != self.column):
            raise TypeError("The matrix must be square for decomposition")

        A = copy.deepcopy(self.matrix)

        for k in range(self.row-1):
            for i in range (k+1, self.row):

                if (abs(A[k][k]) <= eps):
                    raise ValueError('LU decomposition - Cannot divide with zero')

                A[i][k] /= A[k][k]
                for j in range(k+1, self.row):
                    A[i][j] -= A[i][k]*A[k][j]

        return Matrix(array=A)


    def lupDecomposition(self):
        A = copy.deepcopy(self.matrix)
        P = [i for i in range(self.row)]
        
        eps = 10**(-9)

        for i in range(self.row-1):
            pivot = i
            for j in range(i+1, self.row):
                if (abs(A[P[j]][i]) > abs(A[P[pivot]][i])):
                    pivot = j
            P[i], P[pivot] = P[pivot], P[i]

            for j in range(i+1,self.row):
                if (abs(A[P[i]][i]) <= eps):
                    raise ValueError('LUP decomposition - Cannot divide with zero')
                A[P[j]][i] /= A[P[i]][i]
                for k in range(i+1,self.row):
                    A[P[j]][k] -= A[P[j]][i]*A[P[i]][k]

        A = Matrix.swapRows(A, P)
        return Matrix(array=A),P


    @staticmethod
    def swapRows(A, P):
        new_matrix = []
        for i in P:
            new_matrix.append(A[i])
        return new_matrix
    
    @staticmethod
    def identity(size):
        return  Matrix(array=[[1 if i==j else 0 for j in range(size)] for i in range(size)])


    def getUpper(self):
        upper_matrix = copy.deepcopy(self.matrix)

        for i in range(self.row):
            for j in range(self.row):
                if j<i:
                    upper_matrix[i][j] = 0
    
        return Matrix(array=upper_matrix)


    def getLower(self):
        lower_matrix = copy.deepcopy(self.matrix)

        for i in range(self.row):
            for j in range(self.row):
                if i==j:
                    lower_matrix[i][j] = 1
                if j>i:
                    lower_matrix[i][j] = 0

        return Matrix(array=lower_matrix)


    def forwardSubstitution(self, b_vector):
        y = self.row * [0]
        for i in range(self.row):
            y[i] = b_vector[i]
            for j in range(self.row):
                if (i==j):
                    continue
                y[i] -= self.matrix[i][j]*y[j]

        return y


    def backwardSubstitution(self,y_vector):
        
        eps = 10**(-9)

        for i in reversed(range(0,self.row)):
            if abs(self.matrix[i][i]) < eps:
                raise ZeroDivisionError         
            y_vector[i] /= self.matrix[i][i]
            for j in range(i):
                y_vector[j] -= self.matrix[j][i]*y_vector[i]

        return y_vector


    def solveLUP(self, b_vector):

        print("\nSolution with LUP:")
        LU,P = self.lupDecomposition()
        b_vector = Matrix.swapRows(b_vector,P)        
        U = LU.getUpper()
        print("\nMatrix U:")
        U.print()
        L = LU.getLower()
        print("Matrix L:")
        L.print()

        y = L.forwardSubstitution(b_vector)
        print("Vector y:", y)
        x = U.backwardSubstitution(y)
        print("Vector x:", x)


    def solveLU(self, b_vector):

        print("\nSolution with LU:")
        LU = self.luDecomposition()
        U = LU.getUpper()
        print("\nMatrix U:")
        U.print()
        L = LU.getLower()
        print("Matrix L:")
        L.print()

        y = L.forwardSubstitution(b_vector)
        x = U.backwardSubstitution(y)
        print("Vector x:", x)


    def inverse(self):
        
        inv = []

        A = self.clone()
        LU, P = A.lupDecomposition()
        U = LU.getUpper()
        L = LU.getLower()

        E = Matrix.identity(self.row)
        E.matrix = Matrix.swapRows(E.matrix,P)
        E.transpone()

        for e in E.matrix:
            y = L.forwardSubstitution(e)
            try:
                x = U.backwardSubstitution(y)
            except ZeroDivisionError:
                print('Matrix is singular!')
                quit()
            inv.append(x)
        
        return Matrix(array=inv).transpone()
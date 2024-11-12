from petsc4py import PETSc

# Configurar el comunicador global para paralelización
comm = PETSc.COMM_WORLD
size = comm.getSize()  # Número total de procesos
rank = comm.getRank()  # Identificador del proceso actual

# Dimensión del sistema (por ejemplo, 1000 x 1000)
n = 1000
A = PETSc.Mat().createAIJ([n, n], comm=comm)  # Matriz dispersa en paralelo
A.setFromOptions()
A.setUp()

start, end = A.getOwnershipRange() # Rango de filas que maneja cada proceso

print(start, end)

# Definir la matriz laplaciana tridiagonal en paralelo
for i in range(start, end):
    A.setValue(i, i, 2.0)  # Diagonal principal
    if i > 0:
        A.setValue(i, i - 1, -1.0)  # Subdiagonal
    if i < n - 1:
        A.setValue(i, i + 1, -1.0)  # Superdiagonal

A.assemble() # Ensamblar la matriz para usarla en el solver

# Configuración del solucionador Krylov y el precondicionador
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('cg')  # Método de Gradientes Conjugados
ksp.getPC().setType('asm')  # Metodo Aditivo de Schwarz como precondicionador

#PAx= #Pb

# Crear los vectores en paralelo
x, b = A.getVecs()
x.set(0)  # Vector solución inicializado en cero
b.set(1)  # Vector fuente b lleno de unos

# Resolver el sistema en paralelo
ksp.solve(b, x)

print('convergencia en ',ksp.getIterationNumber(), ' iteraciones')
# Recoger y mostrar el resultado en el proceso 0
if rank == 0:
    x_array = x.getArray()
    print("Solución en el proceso 0:")
    print(x_array[:10])  # Imprimir primeros 10 valores de la solución
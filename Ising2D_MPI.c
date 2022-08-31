#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define L 768 //lado da matriz 
#define IT 1152 //numero de interacoes 
#define EQ 154 // equilibra o sistema 

#define nome "XYXY"

int **Inicializacao(int **A) 
{
  int i, j, x;
  srand(time(NULL) );
  for(i=0; i<L; i++)
  {
		for(j=0; j<L; j++)
    {
      x = -1+rand() % 3;

			if(x == 0)
      {
				A[i][j] = -1;
      }
      else 
				A[i][j] = 1;
    }
  }
	return A;
}

int **Contorno(int **A, int **B)
{
  int i, j;

  for(j=1; j<=L; j++)
  {
    B[0][j] = A[L-1][j-1];
    B[L+1][j] = A[0][j-1];
    B[j][0] = A[j-1][L-1];
    B[j][L+1] = A[L-1][0];
  }
  
  for(i=1; i<=L; i++)
  {
    for(j=1; j<=L; j++)
    {
      B[i][j] = A[i-1][j-1];
    }
  }

  return B;
}

double Energia(int **B) 
{
  double en = 0, E = 0;
  int i, j;

  for(i=1; i<L+1; i++)
  {
		for(j=1; j<L+1; j++)
		{	
			en = B[i][j]*(B[i-1][j]+B[i][j+1]+B[i+1][j]+B[i][j-1]);
			E += -en;
		}
  }
	E = E/(L*L);		
  return E/2;
}

double Magnetizacao(int **B)
{
  double total = 0;
  int i, j, N;

  N = L*L;

	for(i=1; i<L+1; i++)
  {
		for(j=1; j<L+1; j++)
    {
			total += B[i][j];
    }
  }
  return total/N;
}

void monteCarlo(int **B, double T)
{
  int i, j, a, b;
  double dE, prob, random;  

  for(i=1; i<L+1; i++)
  {
    for(j=1; j<L+1; j++)
    {
      a = 1+rand() % L;
      b = 1+rand() % L;

      dE = 2*B[a][b]*(B[a-1][b]+B[a][b+1]+B[a+1][b]+B[a][b-1]);

      if(dE<=0)
      {
        B[a][b] *= -1;
      }
      else
      {
        prob = exp(-dE/T);
        random = drand48();
	
        if(prob > random)
        {
          	B[a][b]*=-1;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int i, j, q, size, rank, pontos, quant;
  int **A, **B, random, a, b, contagem=0, imp;
  double prob, dE, Ei, Ef, mag, Ti;
  double *temp, *ptemp, *magIni, *energiaIni, *aux, *magneto, *energetic, *magnetoAux, *energeticAux;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  A = malloc(L*sizeof(int*));
  for(i=0;i<L;i++) 
  {
    A[i] = (int*)malloc(L*sizeof(int));
  }

  B = malloc((L+2)*sizeof(int*));
  for(i=0;i<(L+2);i++) 
  {
    B[i] = (int*)malloc((L+2)*sizeof(int));
  }
 
  //definir a quantidade de elementos do vetor da temperatura:
  //T_final - T_inicial = DeltaT 
  //pontos = DeltaT/passos -> pontos = (5.0 - 0.0)/0.05
  pontos = 100; 

  while(pontos%size != 0)
  {
	  pontos += 1;
	  contagem += 1;
  }

  //aloca o vetor temperatura
  temp = (double *)malloc(pontos*sizeof(double));

  //aloca os vetores iniciais
  magIni = (double *)calloc(pontos, sizeof(double));
  energiaIni = (double *)calloc(pontos, sizeof(double));

  //define o tamanho dos elementos do vetor de pedacinhos:
  quant = pontos/size;

  //aloca o vetor que irao receber a temperatura
  ptemp = (double *)malloc(pontos*sizeof(double));

  //aloca os vetores de magnetizacao e energia
  magneto = (double *)malloc(pontos*sizeof(double));
  energetic = (double *)malloc(pontos*sizeof(double));

  //preenche o vetor da temperatura
  Ti = 0.0;
  for(i=0; i<pontos; i++)
  {
	  temp[i]=Ti;
	  Ti+=0.05;
  }

  //aloca os vetores auxiliares
  aux = (double *)malloc(pontos*sizeof(double));
  magnetoAux = (double *)malloc(pontos*sizeof(double));
  energeticAux = (double *)malloc(pontos*sizeof(double));

  //barreira + espalha os elementos dos vetores para os processadores
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatter(temp, quant, MPI_DOUBLE, ptemp, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(magIni, quant, MPI_DOUBLE, magneto, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(energiaIni, quant, MPI_DOUBLE, energetic, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  srand(time(NULL)+rank);
  for(q=0; q<quant; q++)
  {
    A = Inicializacao(A);
    B = Contorno(A, B);
    
    for(j=0; j<EQ; j++)
    { // equilibra o sistema
      monteCarlo(B, ptemp[q]);
    }

    mag = 0.0;
    Ef = 0.0;
    for(i=0; i<IT; i++)
    { 
      monteCarlo(B, ptemp[q]);  
      mag += Magnetizacao(B);
      magneto[q] += mag;
      //calcula a energia do novo estado
      Ef += Energia(B);
      energetic[q] += Ef;
    }
  }

  //barreira para organizar
  MPI_Barrier(MPI_COMM_WORLD);

  //master recebe os resultados dos processadores para:
  //temperatura
  MPI_Gather(ptemp, quant, MPI_DOUBLE, aux, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //magnetizacao
   MPI_Gather(magneto, quant, MPI_DOUBLE, magnetoAux, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //energia
  MPI_Gather(energetic, quant, MPI_DOUBLE, energeticAux, quant, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //salva os dados em arquivos
  if(rank == 0)
  {     
    imp = pontos - contagem;
    FILE *arq = fopen(nome, "wt");
    fprintf(arq, "temperatura\t magnetizacao\t energia\t\n");
    if(arq ==  NULL)
    {
      printf("\nErro\n");
      exit(1);
    }

    for(i=0; i<imp; i++)
    {
      fprintf(arq, "%6.2f\t %6.2f\t %6.2f\t\n", aux[i], magnetoAux[i]/IT, energeticAux[i]/IT); 
    }

    //libera geral
  	fclose(arq);
  	free(A);
  	free(B);
  	free(temp);
  	free(ptemp);
  	free(aux);
  	free(magneto);
    free(energetic);
    //finalizou mesmo?
    //printf("\nThat's all folks\n");
  }

  MPI_Finalize();

  return 0;
}


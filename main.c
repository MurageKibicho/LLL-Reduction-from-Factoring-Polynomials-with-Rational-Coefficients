#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <assert.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpq.h>
#include <mpfr.h>
#include "LatticeDeets.h"
#define INDEX_2D(i, j, cols) ((i) * (cols) + (j))
//clear && gcc main.c -o m.o -lm -lgmp -lmpfr -lflint -lopenblas && ./m.o
fmpz_t *StringArrayToLattice(int stringBase, size_t rowCount, size_t colCount, char *matrixString[])
{
	fmpz_t *result = malloc(rowCount * colCount * sizeof(fmpz_t));
	for(size_t i = 0; i < rowCount; i++)
	{
		for(size_t j = 0; j < colCount; j++)
		{
			size_t index = INDEX_2D(i,j,colCount);
			fmpz_init(result[index]);
			fmpz_set_str(result[index], matrixString[index], stringBase);
		}
	}
	return result;
}

void PrintLattice(size_t rowCount, size_t colCount, fmpz_t *lattice)
{
	for(size_t i = 0; i < rowCount; i++)
	{
		printf("[");
		for(size_t j = 0; j < colCount; j++)
		{
			size_t index = INDEX_2D(i,j,colCount);
			fmpz_print(lattice[index]);
			if(j + 1 != colCount){printf(", ");} 
		}
		printf("]\n");
	}
	printf("\n");
}

void PrintMPFRMatrix(size_t rowCount, size_t colCount, mpfr_t *matrix)
{
	for(size_t i = 0; i < rowCount; i++)
	{
		printf("[");
		for(size_t j = 0; j < colCount; j++)
		{
			size_t index = INDEX_2D(i,j,colCount);
			mpfr_printf("%.10Rf", matrix[index]);
			if(j + 1 != colCount){printf(", ");} 

		}
		printf("]\n");
	}
	printf("\n");
}

void CompleteGramSchmidt(size_t rowCount, size_t colCount, mpfr_t *orthogonalVector, mpfr_t *gramCoefficients, fmpz_t *basis)
{
	mpz_t tempMPZ;
	mpz_init(tempMPZ);
	mpfr_t numerator;
	mpfr_t denominator;
	mpfr_t temp;

	mpfr_init2(numerator, 256);
	mpfr_init2(denominator, 256);
	mpfr_init2(temp, 256);
	for(size_t i = 0; i < rowCount; i++)
	{
		/*Copy basis row into orthogonal vector*/
		for(size_t j = 0; j < colCount; j++)
		{	
			size_t index = INDEX_2D(i,j,colCount);
			fmpz_get_mpz(tempMPZ, basis[index]);  
			mpfr_set_z(orthogonalVector[index], tempMPZ, MPFR_RNDN);
		}
		/*Orthogonalize against previous vectors*/
		for(size_t j = 0; j < i; j++)
		{
			mpfr_set_ui(numerator, 0, MPFR_RNDN);
			mpfr_set_ui(denominator, 0, MPFR_RNDN);
			/*numerator = <b_i, b*_j>*/
			for(size_t k = 0; k < colCount; k++)
			{
				size_t basisIndex = INDEX_2D(i, k, colCount);
				size_t orthIndex  = INDEX_2D(j, k, colCount);
				fmpz_get_mpz(tempMPZ, basis[basisIndex]);
				mpfr_set_z(temp, tempMPZ, MPFR_RNDN);
				mpfr_mul(temp, temp,orthogonalVector[orthIndex],MPFR_RNDN);
				mpfr_add(numerator, numerator, temp, MPFR_RNDN);
			}

			/*denominator = <b*_j, b*_j> */
			for(size_t k = 0; k < colCount; k++)
			{
				size_t orthIndex = INDEX_2D(j, k, colCount);
				mpfr_mul(temp,orthogonalVector[orthIndex],orthogonalVector[orthIndex],MPFR_RNDN);
				mpfr_add(denominator, denominator, temp, MPFR_RNDN);
			}
			/*mu[i,j] = numerator / denominator*/
			size_t muIndex = INDEX_2D(i, j, rowCount);
			mpfr_div(gramCoefficients[muIndex],numerator,denominator,MPFR_RNDN);

			/*b*_i -= mu[i,j] * b*_j */
			for(size_t k = 0; k < colCount; k++)
			{
				size_t current = INDEX_2D(i, k, colCount);
				size_t previous = INDEX_2D(j, k, colCount);
				mpfr_mul(temp,gramCoefficients[muIndex],orthogonalVector[previous],MPFR_RNDN);
				mpfr_sub(orthogonalVector[current],orthogonalVector[current],temp,MPFR_RNDN);
			}
		}
	}
	mpz_clear(tempMPZ);
	mpfr_clear(numerator);
	mpfr_clear(denominator);
	mpfr_clear(temp);
}

int CheckOrthogonality(size_t rowCount,size_t colCount,mpfr_t *orthogonalVector,double tolerance)
{
	mpfr_t dot,tmp;
	mpfr_init2(dot,256);
	mpfr_init2(tmp,256);
	int result = rowCount+1;
	for(size_t i=0;i<rowCount;i++)
	{
		for(size_t j=i+1;j<rowCount;j++)
		{
			mpfr_set_zero(dot,0);
			for(size_t k=0;k<colCount;k++)
			{
				size_t idx_i=INDEX_2D(i,k,colCount);
				size_t idx_j=INDEX_2D(j,k,colCount);
				mpfr_mul(tmp,orthogonalVector[idx_i],orthogonalVector[idx_j],MPFR_RNDN);
				mpfr_add(dot,dot,tmp,MPFR_RNDN);
			}

			double val=mpfr_get_d(dot,MPFR_RNDN);
			if(fabs(val)>tolerance){result = j; i = rowCount + 1;break;}
		}
	}
	mpfr_clear(dot);
	mpfr_clear(tmp);
	return result;
}

void TestGramSchmidt()
{
	size_t rowCount = rowCountLattice;
	size_t colCount = colCountLattice;
	int stringBase = 10;
	size_t matrixStringLength = sizeof(matrixString) / sizeof(matrixString[0]);
	assert(matrixStringLength == rowCount * colCount);
	fmpz_t prime,errorBound;
	fmpz_init(prime);fmpz_init(errorBound);
	fmpz_set_str(prime, prime_str, 10);
	fmpz_set_str(errorBound, errorBoundString, 10);
	fmpz_t *lattice = StringArrayToLattice(stringBase, rowCount, colCount, matrixString);
	mpfr_t *orthogonalVector = malloc(rowCount * colCount * sizeof(mpfr_t));
	mpfr_t *gramCoefficients = malloc(rowCount * rowCount * sizeof(mpfr_t));
	/*Initialize orthogonalVector*/
	for(size_t i = 0; i < rowCount * colCount; i++)
	{
		mpfr_init2(orthogonalVector[i], 256);  
		mpfr_set_zero(orthogonalVector[i], 0);
	}

	/*Initialize gramCoefficients*/
	for(size_t i = 0; i < rowCount * rowCount; i++)
	{
		mpfr_init2(gramCoefficients[i], 256);
		mpfr_set_zero(gramCoefficients[i], 0);
	}
	printf("matrixStringLength: %ld\n", matrixStringLength);
	printf("Prime:");fmpz_print(prime);printf("\n");
	printf("ErrorBound:");fmpz_print(errorBound);printf("\n");
	
	PrintLattice(rowCount, colCount,lattice);
	CompleteGramSchmidt(rowCount, colCount, orthogonalVector, gramCoefficients, lattice);
	int orthogonalPass = CheckOrthogonality(rowCount, colCount,orthogonalVector,1e-9);
	PrintMPFRMatrix(rowCount, colCount,orthogonalVector);
	PrintMPFRMatrix(rowCount, rowCount,gramCoefficients);
	
	assert(orthogonalPass == rowCount + 1);
	for(size_t i = 0; i < rowCount * colCount; i++){mpfr_clear(orthogonalVector[i]);}
	for(size_t i = 0; i < rowCount * rowCount; i++){mpfr_clear(gramCoefficients[i]);}
	for(size_t i = 0; i < rowCount * colCount; i++){fmpz_clear(lattice[i]);}
	free(lattice);free(gramCoefficients);free(orthogonalVector);
	fmpz_clear(prime);fmpz_clear(errorBound);	
}

void SwapRows(fmpz_t *basis,size_t rowCount,size_t colCount,size_t i,size_t j)
{
	for(size_t k=0;k<colCount;k++)
	{
		size_t a=INDEX_2D(i,k,colCount);
		size_t b=INDEX_2D(j,k,colCount);
		fmpz_swap(basis[a],basis[b]);
	}
}

void SizeReduceRow(size_t k,size_t j,size_t rowCount,size_t colCount,fmpz_t *basis,mpfr_t *mu)
{
	mpz_t q;
	mpz_init(q);
	size_t muIndex=INDEX_2D(k,j,rowCount);
	mpfr_get_z(q,mu[muIndex],MPFR_RNDN);

	fmpz_t q_fmpz,tmp;
	fmpz_init(q_fmpz);
	fmpz_init(tmp);
	fmpz_set_mpz(q_fmpz,q);

	for(size_t c=0;c<colCount;c++)
	{
		size_t ik=INDEX_2D(k,c,colCount);
		size_t ij=INDEX_2D(j,c,colCount);

		fmpz_mul(tmp,basis[ij],q_fmpz);
		fmpz_sub(basis[ik],basis[ik],tmp);
	}

	fmpz_clear(q_fmpz);
	fmpz_clear(tmp);
	mpz_clear(q);
}
void LLLReduction(size_t rowCount,size_t colCount,fmpz_t *basis,fmpz_t *reducedBasis,double delta)
{
	for(size_t i=0;i<rowCount*colCount;i++)fmpz_set(reducedBasis[i],basis[i]);
	mpfr_t *Bstar=malloc(rowCount*colCount*sizeof(mpfr_t));
	mpfr_t *mu=malloc(rowCount*rowCount*sizeof(mpfr_t));
	for(size_t i=0;i<rowCount*colCount;i++)mpfr_init2(Bstar[i],256);
	for(size_t i=0;i<rowCount*rowCount;i++)mpfr_init2(mu[i],256);

	mpfr_t normK,normK1,lovasz,left,tmp;
	mpfr_init2(normK,256);
	mpfr_init2(normK1,256);
	mpfr_init2(lovasz,256);
	mpfr_init2(left,256);
	mpfr_init2(tmp,256);

	CompleteGramSchmidt(rowCount,colCount,Bstar,mu,reducedBasis);
	size_t k=1;
	while(k<rowCount)
	{
		for(ssize_t j=(ssize_t)k-1;j>=0;j--)
		{
			size_t muIndex=INDEX_2D(k,j,rowCount);

			if(mpfr_cmp_d(mu[muIndex],0.5)>0 || mpfr_cmp_d(mu[muIndex],-0.5)<0)
			{
				SizeReduceRow(k,j,rowCount,colCount,reducedBasis,mu);
				CompleteGramSchmidt(rowCount,colCount,Bstar,mu,reducedBasis);
			}
		}

		mpfr_set_zero(normK,0);
		for(size_t c = 0; c < colCount; c++)
		{
			size_t idx=INDEX_2D(k,c,colCount);
			mpfr_mul(tmp,Bstar[idx],Bstar[idx],MPFR_RNDN);
			mpfr_add(normK,normK,tmp,MPFR_RNDN);
		}

		mpfr_set_zero(normK1,0);
		for(size_t c = 0; c < colCount; c++)
		{
			size_t idx=INDEX_2D(k-1,c,colCount);
			mpfr_mul(tmp,Bstar[idx],Bstar[idx],MPFR_RNDN);
			mpfr_add(normK1,normK1,tmp,MPFR_RNDN);
		}

		size_t muIndex=INDEX_2D(k,k-1,rowCount);

		mpfr_mul(lovasz,mu[muIndex],mu[muIndex],MPFR_RNDN);
		mpfr_mul(lovasz,lovasz,normK1,MPFR_RNDN);
		mpfr_add(lovasz,lovasz,normK,MPFR_RNDN);

		mpfr_mul_d(left,normK1,delta,MPFR_RNDN);

		if(mpfr_cmp(lovasz,left)>=0)k++;
		else
		{
			SwapRows(reducedBasis,rowCount,colCount,k,k-1);
			CompleteGramSchmidt(rowCount,colCount,Bstar,mu,reducedBasis);
			k=(k>1)?k-1:1;
		}
	}

	mpfr_clear(normK);
	mpfr_clear(normK1);
	mpfr_clear(lovasz);
	mpfr_clear(left);
	mpfr_clear(tmp);

	for(size_t i=0;i<rowCount*colCount;i++)mpfr_clear(Bstar[i]);
	for(size_t i=0;i<rowCount*rowCount;i++)mpfr_clear(mu[i]);
	free(Bstar);
	free(mu);
}


void TestLLL()
{
	size_t rowCount = rowCountLattice;
	size_t colCount = colCountLattice;
	int stringBase = 10;
	size_t matrixStringLength = sizeof(matrixString) / sizeof(matrixString[0]);
	assert(matrixStringLength == rowCount * colCount);
	fmpz_t prime,errorBound;
	fmpz_init(prime);fmpz_init(errorBound);
	fmpz_set_str(prime, prime_str, 10);
	fmpz_set_str(errorBound, errorBoundString, 10);
	fmpz_t *lattice = StringArrayToLattice(stringBase, rowCount, colCount, matrixString);
	
	fmpz_t *reduced = malloc(rowCount * colCount * sizeof(fmpz_t));

	for(size_t i = 0; i < rowCount * colCount; i++)
	{
		fmpz_init(reduced[i]);
	}

	printf("matrixStringLength: %ld\n", matrixStringLength);
	printf("Prime:");fmpz_print(prime);printf("\n");
	printf("ErrorBound:");fmpz_print(errorBound);printf("\n");
	
	PrintLattice(rowCount, colCount,lattice);
	LLLReduction(rowCount, colCount,lattice,reduced, 0.75);
	PrintLattice(rowCount, colCount,reduced);
	
	for(size_t i = 0; i < rowCount * colCount; i++){fmpz_clear(lattice[i]);fmpz_clear(reduced[i]);}
	free(lattice);
	free(reduced);
	fmpz_clear(prime);fmpz_clear(errorBound);	
}
int main()
{
	//TestGramSchmidt();
	TestLLL();
	flint_cleanup();
	return 0;
}

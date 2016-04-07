#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
//#define HW_BASE 0xfc000000
//#define HW_SPAN 0x04000000
//#define ALT_LWFPGASLVS_OFST     0xff200000

#define HW_BASE 0xc0000000
#define HW_SPAN 0x40000000
#define HW_MASK HW_SPAN - 1

double ** mult(double **A, double **x, int dim1, int dim2, int dim3)
{
//	printf("%p\n", A);
//	printf("%p\n", A[0]);
//	printf("%p\n", &(A[0][0]));
//	printf("%p\n", B);
	int i, j, k;
	double sum = 0;
	double **ret = (double**)malloc(dim1 * sizeof(double*));
	for (i = 0; i < dim1; i++) {
		ret[i] = (double*)malloc(dim3 * sizeof(double));
	}
	
	for (i = 0; i < dim1; i++) {
		for (j = 0; j < dim3; j++) {
			for (k = 0; k < dim2; k++) {
				sum += A[i][k] * x[k][j];
				//printf("a = %.4lf, b = %.4lf\n", A[i][k] , x[k][j]);
			}
			ret[i][j] = sum;
			sum = 0;
		}
	}

	return ret;
}

double ** add(double **A, double **B, int dim1, int dim2)
{
	int i, j;
	double **ret = (double**)malloc(dim1 * sizeof(double*));
	for (i = 0; i < dim1; i++) {
		ret[i] = (double*)malloc(dim2 * sizeof(double));
	}
	//printf("a = %.2lf, b = %.2lf\n", **A , **B);
//	printf("%p\n", A);
//	printf("%p\n", A[0]);
//	printf("%p\n", &(A[0][0]));
//	printf("%p\n", B);
	for (i = 0; i < dim1; i++) {
		for (j = 0; j < dim2; j++) {
			ret[i][j] = A[i][j] + B[i][j];			
			//printf("a = %.2lf, b = %.2lf\n", A[i][j] , B[i][j]);
		}
	}
	
	return ret;
}


void sigmoid(double **A, int dim1)
{
	int i, j;
	for (i = 0; i < dim1; i++) {
		A[i][0] = 1.0/(1.0 + exp(-A[i][0]));
	} 
}

#define B1L1_OFFSET	2
#define B1L2_OFFSET	(B1L1_OFFSET + 200)
#define W1L1_OFFSET	(B1L2_OFFSET + 200)
#define W1L2_OFFSET	(W1L1_OFFSET + 200 * 784)
#define SOFTMAX_OFFSET	(W1L2_OFFSET + 200 * 200)
#define DATA_OFFSET	(SOFTMAX_OFFSET + 10 * 200)
#define LABELS_OFFSET	(DATA_OFFSET + 1000 * 784)

int main(void)
{
	FILE *finalB1L1 = fopen("finalB1L1.txt", "r");
	FILE *finalB1L2 = fopen("finalB1L2.txt", "r");
	FILE *finalW1L1 = fopen("finalW1L1.txt", "r");
	FILE *finalW1L2 = fopen("finalW1L2.txt", "r");
	FILE *finalSoftmaxTheta = fopen("finalSoftmaxTheta.txt", "r");
	FILE *testData = fopen("testData.txt", "r");
	FILE *testLabels = fopen("testLabels.txt", "r");

	double *B1L1[200];
	double *B1L2[200];
	double *W1L1[200];
	double *W1L2[200];
	double *SoftmaxTheta[10];
	double **Data[10000];
	int Labels[10000];

	char line[8000];
	char *ptr;
	int i;
	int j;


	void *VA;
	void* sdram;
	int fd;
	clock_t time;


	if ((fd = open("/dev/mem", (O_RDWR|O_SYNC))) == -1) {
		perror("ERROR: could not open \"/dev/mem\"\n");
		return 1;
	}

	VA = mmap(NULL, HW_SPAN, (PROT_READ|PROT_WRITE), MAP_SHARED, fd, HW_BASE);
	if (VA == MAP_FAILED) {
		perror("ERROR: mmap() failed ... \n");
		close(fd);
		return 1;
	}
	sdram = VA + ((unsigned long)(HW_BASE + 0x00) & (unsigned long)(HW_MASK));
	printf("sdram = 0x%X\n", (unsigned int)sdram);






	puts("malloc start");
	for (i=0; i<200; i++) {
		B1L1[i] = (double*)malloc(1*sizeof(double));
		B1L2[i] = (double*)malloc(1*sizeof(double));
		W1L1[i] = (double*)malloc(784*sizeof(double));
		W1L2[i] = (double*)malloc(200*sizeof(double));
	}

	for (i=0; i<10; i++) {
		SoftmaxTheta[i] = (double*)malloc(200*sizeof(double));
	}

	for (i=0; i<10000; i++) {
		Data[i] = (double**)malloc(784*sizeof(double*));
		for (j=0; j<784; j++) {
			Data[i][j] = (double*)malloc(1*sizeof(double));
		}
	}
	puts("malloc end");





	i = 0;
	while (fgets(line, 8000, finalB1L1) != NULL) {
		line[strlen(line)-1] = '\0';
		B1L1[i][0] = atof(line);
		i++;
	}


	i = 0;
	while (fgets(line, 8000, finalB1L2) != NULL) {
		line[strlen(line)-1] = '\0';
		B1L2[i][0] = atof(line);
		i++;
	}

	i = 0;
	while (fgets(line, 8000, testLabels) != NULL) {
		line[strlen(line)-1] = '\0';
		Labels[i] = atoi(line);
		i++;
	}

	i = 0;
	j = 0;
	while (fgets(line, 8000, finalW1L1) != NULL) {
		line[strlen(line)-1] = '\0';
		ptr = strtok(line, ",");

		while (ptr != NULL) {
			W1L1[i][j] = atof(ptr);
			j++;
			ptr = strtok(NULL, ",");
		}
		j = 0;
		i++;
	}

	i = 0;
	j = 0;
	while (fgets(line, 8000, finalW1L2) != NULL) {
		line[strlen(line)-1] = '\0';
		ptr = strtok(line, ",");

		while (ptr != NULL) {
			W1L2[i][j] = atof(ptr);
			j++;
			ptr = strtok(NULL, ",");
		}
		j = 0;
		i++;
	}

	i = 0;
	j = 0;
	while (fgets(line, 8000, finalSoftmaxTheta) != NULL) {
		line[strlen(line)-1] = '\0';
		ptr = strtok(line, ",");

		while (ptr != NULL) {
			SoftmaxTheta[i][j] = atof(ptr);
			j++;
			ptr = strtok(NULL, ",");
		}
		j = 0;
		i++;
	}

	i = 0;
	j = 0;
	while (fgets(line, 8000, testData) != NULL) {
		line[strlen(line)-1] = '\0';
		ptr = strtok(line, ",");

		while (ptr != NULL) {
			Data[i][j][0] = atof(ptr);
			j++;
			ptr = strtok(NULL, ",");
		}
		j = 0;
		i++;
	}
	puts("init done");

	//puts("SDRAM begin");
	time = (float) clock();

	for (i = 0; i<200; i++) {
		((double*)sdram)[B1L1_OFFSET+i] = B1L1[i][0];
		//B1L1[i][0] = ((double*)sdram)[B1L1_OFFSET+i];
		B1L1[i] = &((double*)sdram)[B1L1_OFFSET+i];
		//printf("%.3lf\n", ((double*)sdram)[B1L1_OFFSET+i]);
		//printf("%.3lf\n", B1L1[i][0]);
		//puts("");
	}

	for (i = 0; i<200; i++) {
		((double*)sdram)[B1L2_OFFSET+i] = B1L2[i][0];
		B1L2[i] = &((double*)sdram)[B1L2_OFFSET+i];
	}

	for (i = 0; i<200; i++) {
		for (j = 0; j<784; j++) {
			((double*)sdram)[W1L1_OFFSET+i*784+j] = W1L1[i][j];
			//W1L1[i][j] = ((double*)sdram)[W1L1_OFFSET+i*784+j];
		}
		W1L1[i] = &((double*)sdram)[W1L1_OFFSET+i*784];
	}

	for (i = 0; i<200; i++) {
		for (j = 0; j<200; j++) {
			((double*)sdram)[W1L2_OFFSET+i*200+j] = W1L2[i][j];
			//W1L2[i][j] = ((double*)sdram)[W1L2_OFFSET+i*200+j];
		}
		W1L2[i] = &((double*)sdram)[W1L2_OFFSET+i*200];
	}

	for (i = 0; i<10; i++) {
		for (j = 0; j<200; j++) {
			((double*)sdram)[SOFTMAX_OFFSET+i*200+j] = SoftmaxTheta[i][j];
			//SoftmaxTheta[i][j] = ((double*)sdram)[SOFTMAX_OFFSET+i*200+j];
		}
		SoftmaxTheta[i] = &((double*)sdram)[SOFTMAX_OFFSET+i*200];
	}

	for (i = 0; i<100; i++) {
		for (j = 0; j<784; j++) {
			((double*)sdram)[DATA_OFFSET+i*784+j] = Data[i][j][0];
			Data[i][j] = &((double*)sdram)[DATA_OFFSET+i*784+j];
		}
	}
	//for (i = 0; i<10000; i++) {
	//	((int*)sdram)[LABELS_OFFSET+i] = Labels[i];
	//	Labels[i] = ((int*)sdram)[LABELS_OFFSET+i];
	//}

//	puts("SDRAM end");
	fprintf(stderr, "SDRAM: %.2f seconds\n",(float) (clock() - time) / CLOCKS_PER_SEC);


//	//double A[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};
//	//double B[3][3] = {{1,1,1}, {1,1,1}, {1,1,1}};
//	double **A = (double**)malloc(3 * sizeof(double*));
//	double **B = (double**)malloc(3 * sizeof(double*));
//	int tmp = 1;
//	for (i = 0; i<3; i++) {
//		A[i] = (double*)malloc(3*sizeof(double));
//		B[i] = (double*)malloc(3*sizeof(double));
//		for (j = 0; j < 3; j++) {
//			A[i][j] = tmp;
//			B[i][j] = 10 - tmp++;
//		}
//	}
//		
//	printf("%p\n", A);
//	printf("%p\n", A[0]);
//	printf("%p\n", &(A[0][0]));
//	printf("%p\n", B);




	double **hidden1, **hidden2, **final, **tmp;
	double max = 0;
	int maxidx = 0;
	int fignum = 0;
	int count = 0;

	time = (float) clock();
	for (fignum = 0; fignum < 100; fignum++) {
		hidden1 = mult(W1L1, (Data[fignum]), 200, 784, 1);
		tmp = add(hidden1, B1L1, 200, 1);
		free(hidden1);
		hidden1 = tmp;
		sigmoid(hidden1, 200);

		hidden2 = mult(W1L2, hidden1, 200, 200, 1);
		tmp = add(hidden2, B1L2, 200, 1);
		free(hidden2);
		hidden2 = tmp;
		sigmoid(hidden2, 200);

		final = mult(SoftmaxTheta, hidden2, 10, 200, 1);
		sigmoid(final, 10);
		
		for (i = 0; i< 10; i++ ) {
			//printf("%.3f\t", final[i][0]);
			if (final[i][0] > max) {
				max = final[i][0];
				maxidx = i + 1;
			}
		}
		//printf("max_index = %d\n", maxidx);

		if (Labels[fignum] == maxidx) {
			count++;
		}
		
		max = 0;
		maxidx = 0;
		free(hidden1);
		free(hidden2);
		free(final);
	}

	fprintf(stderr, "Calculation: %.2f seconds\n",(float) (clock() - time) / CLOCKS_PER_SEC);
	printf("sample size = %d, accuracy = %f\n", fignum, count / (float)fignum);
	return 0;
}


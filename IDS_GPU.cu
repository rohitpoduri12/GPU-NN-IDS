/******************************************************************************
Group Project name -  
Name - Adithya Beemanapalli(ab4348), Rohit Poduri(np2581)

/******************************************************************************
                            D E C L A R A T I O N S
 ******************************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef int           BOOL;
typedef int           INT;
typedef double        REAL;

#define FALSE         0
#define TRUE          1

#define BIAS          1
#define ARRAY_WIDTH   42
#define NUM_EPOCHS    10

#define sqr(x)        ((x)*(x))

int count = 0;
int correct_predict=0;


typedef struct {                     /* A LAYER OF A NET:                     */
        INT           Units;         /* - number of units in this layer       */
        REAL*         Output;        /* - output of ith unit                  */
        REAL*         Error;         /* - error term of ith unit              */
        REAL*        Weight;        /* - connection weights to ith unit      */
        REAL*        WeightSave;    /* - saved weights for stopped training  */
        REAL*        dWeight;       /* - last weight deltas for momentum     */
} LAYER;

typedef struct {                     /* A NET:                                */
        LAYER**       Layer;         /* - layers of this net                  */
        LAYER*        InputLayer;    /* - input layer                         */
        LAYER*        OutputLayer;   /* - output layer                        */
        REAL          Alpha;         /* - momentum factor                     */
        REAL          Eta;           /* - learning rate                       */
        REAL          Gain;          /* - gain of sigmoid function            */
        REAL          Error;         /* - total net error                     */
} NET;

#define BLOCK_SIZE 64


__global__ void
matrixMul_tiling( double* C, double* A, double* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory for A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Shared memory for B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of shared memory A's first element
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of shared memory A's last element
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate A's shared memory
    int aStep  = BLOCK_SIZE;

    // Index of shared memory B's first element
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate B's shared memory
    int bStep  = BLOCK_SIZE * wB;

    //Compute partial result for C
    float Csub = 0;

    // Loop over all the shared memory blocks of A and B
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {


        //Access shared memory
	As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * tx + ty];

        // Wait for all threads to synchronize
        __syncthreads();

        // Continue computing the partial sum
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize threads
        __syncthreads();
    }

    // Write to C matrix
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    	C[c + wB * ty + tx] = Csub;
}


__global__ void
matrixMul_coalescing( double* C, double* A, double* B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int aBegin = wA * BLOCK_SIZE * by;


    int aEnd   = aBegin + wA - 1;

    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;

    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
    {

        //Load shared memory matrices
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[tx][ty] = B[b + wB * ty + tx];

        // Synchronize threads
	        __syncthreads();

        // Compute coalesced sum and add it to the previously computed partial sum
        for (int k = 0; k < BLOCK_SIZE; ++k)
          Csub += As[ty][k] * Bs[tx][k];

        // Synchronize threads
        __syncthreads();
    }

    // Update C matrix
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}




__global__ void
matrixMul_naive( double* C, double* A, double* B, int wA, int wB)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of array
  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  double accumulate = 0.0;

  for(int k=0; k<wA; k++){
    accumulate = accumulate + A[ i * wA + k ] * B[ k * wB + j ];
  }

  // Update C matrix
  C[ i * wB + j ] = accumulate;

}




void computeGold(double* C, const double* A, const double* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}


/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/


void InitializeRandoms()
{
  srand(1171);
}


INT RandomEqualINT(INT Low, INT High)
{
  return rand() % (High-Low+1) + Low;
}      


REAL RandomEqualREAL(REAL Low, REAL High)
{
  return ((REAL) rand() / RAND_MAX) * (High-Low) + Low;
}      


/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/


#define NUM_LAYERS    3
#define N             41
#define M             1
INT                   Units[NUM_LAYERS] = {N, 10, M};



/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/


void GenerateNetwork(NET* Net)
{
  INT l;

  Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));
   
  for (l=0; l<NUM_LAYERS; l++) {
    Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
      
    Net->Layer[l]->Units      = Units[l];
    Net->Layer[l]->Output     = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Error      = (REAL*)  calloc(Units[l]+1, sizeof(REAL));
    Net->Layer[l]->Output[0]  = BIAS;

    if(l != 0){
    Net->Layer[l]->Weight     = (REAL*) calloc((Units[l]+1)*(Units[l-1]+1), sizeof(REAL));
    Net->Layer[l]->WeightSave = (REAL*) calloc((Units[l]+1)*(Units[l-1]+1), sizeof(REAL));
    Net->Layer[l]->dWeight    = (REAL*) calloc((Units[l]+1)*(Units[l-1]+1), sizeof(REAL));
      }

  }
  Net->InputLayer  = Net->Layer[0];
  Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
  Net->Alpha       = 0.9;
  Net->Eta         = 0.25;
  Net->Gain        = 1;

}


void RandomWeights(NET* Net)
{
  INT l,i,j;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[(i*(Net->Layer[l-1]->Units+1))+j] = RandomEqualREAL(-0.5, 0.5);
      }
    }
  }
}


void SetInput(NET* Net, REAL* Input)
{
  INT i;
   
  for (i=1; i<=Net->InputLayer->Units; i++) {
    Net->InputLayer->Output[i] = Input[i-1];
  }
}


void GetOutput(NET* Net, REAL* Output)
{
  INT i;
   
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Output[i-1] = Net->OutputLayer->Output[i];
  }
}



/******************************************************************************
            S T O P P E D   T R A I N I N G
 ******************************************************************************/


void SaveWeights(NET* Net)
{
  INT l,i,j;

  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->WeightSave[(i*(Net->Layer[l-1]->Units+1))+j] = Net->Layer[l]->Weight[(i*(Net->Layer[l-1]->Units+1))+j];
      }
    }
  }
}


void RestoreWeights(NET* Net)
{
  INT l,i,j;
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Net->Layer[l]->Weight[(i*(Net->Layer[l-1]->Units+1))+j] = Net->Layer[l]->WeightSave[(i*(Net->Layer[l-1]->Units+1))+j];
      }
    }
  }
}


/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/


void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper)
{
  INT  i,j;
  double buffer = Upper->Output[0];


    dim3 threads,grid;

    double* d_A;

    double* d_B;
    
    double* d_C;

    unsigned int size_A = (Lower->Units + 1) * (Upper->Units + 1);  	// A is the weight matrix of size (No of upper layer units + 1 ) X (No of lower layer units + 1 )
    unsigned int size_B = 1 * (Lower->Units + 1);			// B is the lower layer input matrix of size (No of lower layer units + 1 ) X (1)
    unsigned int size_C = 1 * (Upper->Units + 1);			// B is the upper layer input matrix of size (No of upper layer units + 1 ) X (1)

    unsigned int mem_size_A = sizeof(double) * size_A;
    unsigned int mem_size_B = sizeof(double) * size_B;
    unsigned int mem_size_C = sizeof(double) * size_C;

    cudaMalloc((void**) &d_A, mem_size_A);				//Initialize cuda memory for the same
    cudaMalloc((void**) &d_B, mem_size_B);
    cudaMalloc((void**) &d_C, mem_size_C);

    int wc = 1;
    int hc = Upper->Units + 1;
    
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);				//Initialize thread block dimensions
    grid = dim3(wc / threads.x, hc / threads.y);			//Initialize grid dimensions

    cudaMemcpy(d_A, Upper->Weight, mem_size_A, cudaMemcpyHostToDevice);		//Copy memory to device

    cudaMemcpy(d_B, Lower->Output, mem_size_B, cudaMemcpyHostToDevice);

    // naive implementation
    //matrixMul_naive<<< grid, threads >>>(d_C, d_A, d_B, Lower->Units + 1, 1);


    //Coalescing implementation
    //matrixMul_coalescing<<< grid, threads >>>(d_C, d_A, d_B, Lower->Units + 1, 1);

    //Tiling implementation
    //matrixMul_tiling<<< grid, threads >>>( d_C, d_A, d_B, Lower->Units + 1, 1);

    cudaMemcpy(Upper->Output, d_C, mem_size_C, cudaMemcpyDeviceToHost); 	// copy result from device to host
   
    Upper->Output[0] = buffer;		//Since the 0th element is a bias element, it should remain the same
   
    for (i=1; i<=Upper->Units; i++) {		// Apply sigmoid function

 	Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Upper->Output[i]));
   }
      
  cudaFree(d_A);		//Free the memories
  cudaFree(d_B);
  cudaFree(d_C);
  
}


void PropagateNet(NET* Net)		//Apply a loop to propogate each layer
{
  INT l;
   
  for (l=0; l<NUM_LAYERS-1; l++) {
    PropagateLayer(Net, Net->Layer[l], Net->Layer[l+1]);
  }
}


/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/


void ComputeOutputError(NET* Net, REAL* Target)
{
  INT  i;
  REAL Out, Err;
   
  Net->Error = 0;
  for (i=1; i<=Net->OutputLayer->Units; i++) {
    Out = Net->OutputLayer->Output[i];
    Err = Target[i-1]-Out;
]
    Net->OutputLayer->Error[i] = Net->Gain * Out * (1-Out) * Err;

    Net->Error += (0.5 * Err * Err);

    if(Net->OutputLayer->Output[i]>0.5)
	Net->OutputLayer->Output[i] = 1;

    else
	Net->OutputLayer->Output[i] = 0;

  }
}


void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower)			//Compute errors for each layer
{
  INT  i,j;
  REAL Out, Err;
   
  for (i=1; i<=Lower->Units; i++) {
    Out = Lower->Output[i];
    Err = 0;
    for (j=1; j<=Upper->Units; j++) {
      Err += Upper->Weight[(i*(Lower->Units+1))+j] * Upper->Error[j];
    }
    Lower->Error[i] = Net->Gain * Out * (1-Out) * Err;
  }
}


void BackpropagateNet(NET* Net)							//Loop through each layer in reverse
{
  INT l;
   
  for (l=NUM_LAYERS-1; l>1; l--) {
    BackpropagateLayer(Net, Net->Layer[l], Net->Layer[l-1]);
  }
}


void AdjustWeights(NET* Net)							//Adjust the weight matrices
{
  INT  l,i,j;
  REAL Out, Err, dWeight;
   
  for (l=1; l<NUM_LAYERS; l++) {
    for (i=1; i<=Net->Layer[l]->Units; i++) {
      for (j=0; j<=Net->Layer[l-1]->Units; j++) {
        Out = Net->Layer[l-1]->Output[j];
        Err = Net->Layer[l]->Error[i];
        dWeight = Net->Layer[l]->dWeight[(i*(Net->Layer[l-1]->Units+1))+j];
        Net->Layer[l]->Weight[(i*(Net->Layer[l-1]->Units+1))+j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
        Net->Layer[l]->dWeight[(i*(Net->Layer[l-1]->Units+1))+j] = Net->Eta * Err * Out;
      }
    }
  }
}


/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/


void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training)		//Compute all the functions for the neural net
{
  SetInput(Net, Input);
  PropagateNet(Net);
  GetOutput(Net, Output);

  ComputeOutputError(Net, Target);
  if (Training) {
    BackpropagateNet(Net);
    AdjustWeights(Net);
  }

  else {

  if(*Target==Net->OutputLayer->Output[1])
     correct_predict++;
  }
  
}


void TrainNet(NET* Net, INT Epochs)				// Train the network
{
  INT  n;	
  REAL Output[M];

  REAL array[ARRAY_WIDTH] ;

  FILE *fp;

  for (n=0; n<Epochs; n++) {

    fp = fopen ("finalkddcup.csv", "r");		//Loop through the file and train the net using each row
    
    int j = 0;
    char *buffer = NULL;
    size_t len = 0;
    ssize_t read;
    char *ptr = NULL;


    while ((read = getline (&buffer, &len, fp)) != -1) 
    {
        int c,i=0;

        for (j = 0, ptr = buffer; j < ARRAY_WIDTH; j++, ptr++) 
	{   
	    array[j] = strtod(ptr, &ptr);		//String to double
	}
        count++;					//Count the number of rows the dataset
	SimulateNet(Net, &(array[0]), Output, &(array[ARRAY_WIDTH - 1]), TRUE);  //Train the net for each row
	}
    }
	
   fclose(fp);		//Close the file

}


void TestNet(NET* Net)
{
  REAL Output[M];

  REAL array[ARRAY_WIDTH] ;

  FILE *fp;


    fp = fopen ("finalkddcup.csv", "r");
    
    int j = 0;
    char *buffer = NULL;
    size_t len = 0;
    ssize_t read;
    char *ptr = NULL;
    int count2 = 0;

    while ((read = getline (&buffer, &len, fp)) != -1) 
    {
        int c,i=0;
        for (j = 0, ptr = buffer; j < ARRAY_WIDTH; j++, ptr++) 
            array[j] = strtod(ptr, &ptr);

	SimulateNet(Net, &(array[0]), Output, &(array[ARRAY_WIDTH - 1]), FALSE);		//Repeat the same process with the exception of backpropogation
    }

}



/******************************************************************************
                                    M A I N
 ******************************************************************************/


int main(int argc, char *argv[])
{
  NET  Net;
  BOOL Stop;

  InitializeRandoms();		//Initialize Random weights for the net
  GenerateNetwork(&Net);	//Generate the Network
  RandomWeights(&Net);		//Assign random weights to the net


  TrainNet(&Net, NUM_EPOCHS);	//Train the neural network with the datapoints
  SaveWeights(&Net);		//Save the trained network
  TestNet(&Net);		//Test the network for accuracy

  return 0;

}


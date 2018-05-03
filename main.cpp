// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "layer.h"
#include "layerlinear.h"
#include "layerconv.h"
#include "layerleakyrect.h"
#include "layermaxpooling2d.h"
#include "layersinusoid.h"
#include "neuralnet.h"
#include <algorithm>
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <fstream>
#include <time.h>
#include "matrix.h"
#include "image.h"


using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;



void testLearner(SupervisedLearner& learner)
{


}


size_t convertToDecimal(const Vec& oneHot)
{
	return oneHot.indexOfMax();
}


void make_features_and_labels(const Matrix& data, Matrix& feats, Matrix& labs)
{

}

void make_training_and_testing(Matrix& trainFeats, Matrix& trainLabs, Matrix& testFeats, Matrix& testLabs, Matrix& data)
{
	size_t trainingDataCount = 256;
	size_t testingDataCount = 100;

	trainFeats.setSize(trainingDataCount, 1);
	trainLabs.setSize(trainingDataCount, 1);
	testFeats.setSize(testingDataCount, 1);
	testLabs.setSize(testingDataCount, 1);

	for(size_t i = 0; i < trainingDataCount; ++i)
	{
		trainFeats[i][0] = (i / (double)trainingDataCount);
		trainLabs[i][0] = data[i][0];

	}
	for(size_t i = 0; i < testingDataCount; ++i)
	{
		testFeats[i][0] = ((i + trainingDataCount) / (double)trainingDataCount);
		testLabs[i][0] = data[trainingDataCount +i][0];
	}



}

void preprocessData(Matrix& m)
{


}

void shuffleTrainData(Matrix& trainFeats, Matrix& trainLabs, Rand random)
{
	if(trainLabs.rows() != trainFeats.rows()) throw Ex("training data not intiialized correctly");
	for(size_t i = 0; i < trainFeats.rows(); ++i)
	{
		size_t r1 = random.next(trainFeats.rows());
		size_t r2 = random.next(trainFeats.rows());
		trainFeats.swapRows(r1,r2);
		trainLabs.swapRows(r1,r2);
	}
}

unsigned int rgbToUint(int r, int g, int b)
{
	return 0xff000000 | ((r & 0xff) << 16) |((g & 0xff) << 8) | (b & 0xff);
}

void makeImage(Vec& state, const char* filename, NeuralNet& nn)
{
	Vec in;
	in.resize(4);
	in[2] = state[0];
	in[3] = state[1];


	MyImage im;
	size_t f = 0;
	if(filename[0] == 't')  f= 3;
	else f = 1;
	size_t w = 64*f;
	size_t h = 48*f;
	im.resize(w, h);
	for(size_t y = 0; y < h; y++)
	{
		in[1] = (double)y / h;
			for(size_t x = 0; x < w; x++)
			{
			  in[0] = (double)x / w;
			  Vec out = nn.predict(in);
				// out.print();
				// cout << endl;
			  unsigned int color =
			    rgbToUint(out[0] * 255, out[1] * 255, out[2] * 255);

			  im.setPixel(x, y, color);
			}
	}
	im.savePng(filename);
}

void preprocessActions(Matrix& A)
{
	for(size_t i = 0; i < A.rows(); ++i)
	{
		for(size_t j = 0; j < A.cols(); ++j)
		{

				if(A[i][j] == 'a') A[i][j] = 0.25;

				else if(A[i][j] == 'b') A[i][j] = 0.5;

				else if(A[i][j] == 'c') A[i][j] = 0.75;

				else if(A[i][j] == 'd') A[i][j] = 1.0;

				else throw Ex("Value of actions data not a,b,c, or d");

		}
	}
}

void unsupervised(Rand random)
{
	// Vec h(5);
	// for(size_t i = 0; i < h.size(); ++i)
	// 	h[i] = i;
  //
	// Vec t(h, 1, 1 +2+1);
	// t.print();


	// NeuralNet nn(random);
	// nn.add(new LayerLinear(1,2));
	// nn.add(new LayerTanh(2));
	// nn.add(new LayerLinear(2,1));
	// nn.init_weights();
	// Vec feat(1);
	// feat.fill(0.3);
	// Vec lab(1);
	// lab.fill(0.7);
	// for(size_t i = 0; i < 3; ++i)
	// {
	// 	nn.predict(feat);
	// 	nn.backprop(lab);
	// 	nn.update_gradient(feat);
	// 	nn.refine_weights(0.1);
	// 	cout << endl;
	// 	cout << "weights at " << i << endl;
	// 	nn.get_weights().print();
	// 	cout << endl;
	// }
	// return;




	string fn = "data/";
	Matrix X;
	X.loadARFF(fn + "observations.arff");
	X *= (1.0/255.0);

	Matrix A;
	A.loadARFF(fn + "actions.arff");
	A.print(cout);
	//preprocessActions(A);


	NeuralNet nn(random);
	nn.add(new LayerLinear(4,12));
	nn.add(new LayerTanh(12));
	nn.add(new LayerLinear(12,12));
	nn.add(new LayerTanh(12));
	nn.add(new LayerLinear(12,3));
	nn.add(new LayerTanh(3));
	nn.init_weights();
	//Matrix trainFeats, trainLabs, testFeats, testLabs;
	nn.train_with_images(X);
	Vec in(2);
	in.fill(0.5);
	makeImage(in,  "test1.png" ,nn);
	makeImage(in,  "ptest1.png" ,nn);
	in[0] = 1.8;
	in[1] = 2;
	makeImage(in,  "test2.png" ,nn);
	makeImage(in,  "ptest2.png" ,nn);

	Matrix V;
	V.copy(nn.getV());
	Matrix newV(V.rows()-1, V.cols());
	for(size_t i = 0; i < newV.rows(); ++i)
	{
		for(size_t j = 0; j < newV.cols(); ++j)
			newV[i][j] = V[i][j];
	}


	//drawing the graph
// 	double x_min = V.columnMin(0) -1.5;
// 	double x_max = V.columnMax(0) +1.5;
//
// 	double y_min = V.columnMin(1) -1.5;
// 	double y_max = V.columnMax(1) +1.5;
//
// 	GSVG svg(1024, 768,x_min, y_min,x_max, y_max);
// 	svg.horizMarks(x_max*10);
// 	svg.vertMarks(y_max);
//
// 	Vec prevPoint = V[0];
//
//
// 	svg.dot(prevPoint[0], prevPoint[1], 1, 0x000080);
// 	cout << "V" << endl;
// 	V.print(cout);
//
// 	for(size_t i = 1 ; i < V.rows(); ++i)
// 	{
// 		Vec currentPoint = V[i];
// 		svg.dot(currentPoint[0], currentPoint[1], 1, 0x000080);
// 		//cout << "Line: " << prevPoint[0] << ", " << prevPoint[1] << "to   " << currentPoint[0] << ", " << currentPoint[1] << endl;
// 		svg.line(prevPoint[0], prevPoint[1], currentPoint[0], currentPoint[1]);
// 		prevPoint.copy(currentPoint);
// 	}
// 	std::ofstream s;
// s.exceptions(std::ios::badbit);
// s.open("a.svg", std::ios::binary);
// svg.print(s);


//adding second MLP
cout << "training xn" << endl;
NeuralNet transition(random);

transition.add(new LayerLinear(3,2));
transition.add(new LayerTanh(2));

double learning_rate =0.1;
for(size_t j = 0; j < 10; ++j)
{
	for(size_t i = 0; i < 10000; ++i)
	{
		for(size_t k = 0; k < newV.rows(); ++k)
		{
			Vec feats(3);
			feats[0] = newV[k][0];
			feats[1] = newV[k][1];
			feats[2] = A[k][0];

			transition.train_incremental(feats, V[k+1]);
			transition.refine_weights(learning_rate);
		}

	}
	cout << j << "/ " << 10 << endl;
	learning_rate*=0.75;
}

size_t v_index = 0;
in.copy(V[v_index]);
v_index++;
makeImage(in,  "frame0.png" ,nn);
for(size_t i = 0; i < 5; ++i)
{
	Vec feats(3);
	feats[0] = V[v_index][0];
	feats[1] = V[v_index][1];
	feats[2] = 0;  //1,0 is a
	in.copy(transition.predict(feats));
	std::string filename = "frame";
	filename.append(to_str(v_index));
	filename.append(".png");
	makeImage(in,  filename.c_str() ,nn);

	v_index++;

}
for(size_t i = 0; i < 5; ++i)
{
	Vec feats(3);
	feats[0] = V[v_index][0];
	feats[1] = V[v_index][1];
	feats[2] = 2.0;
	in.copy(transition.predict(feats));
	std::string filename = "frame";
	filename.append(to_str(v_index));
	filename.append(".png");
	makeImage(in,  filename.c_str() ,nn);
	v_index++;
}







}

void debugSpew(Rand random)
{

}



int main(int argc, char *argv[])
{
	Rand random(5); //50 is solid... 5 is the winner
	enableFloatingPointExceptions();
	int ret = 1;


	try
	{

		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	try
	{
unsupervised(random);




		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}

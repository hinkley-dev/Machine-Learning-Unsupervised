#ifndef NEURALNET_H
#define NEURALNET_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include "layerlinear.h"
#include "layertanh.h"
#include "layerconv.h"
#include "layersinusoid.h"
#include <vector>
#include "supervised.h"
#include "rand.h"
#include <algorithm>
#include <math.h>
#include "svg.h"
#include <time.h>



using namespace std;

class NeuralNet : public SupervisedLearner
{

private:
  Vec weights;
  vector<Layer*> layers;
  Vec gradient;
  Vec inputBlame;
  Rand random;
  Matrix m_V;
  Matrix testingLabels;
  Matrix testingFeatures;
  size_t reg; //0==no reg, 1== l1, 2 ==l2
  const double pi = 3.1415926535898;


public:
  NeuralNet(Rand r);
  ~NeuralNet();
  virtual void train(Matrix& features, Matrix& labels);
  void train(Matrix& trainFeats, Matrix& trainLabs, double learning_rate, size_t batch_size);
	virtual const Vec& predict(const Vec& in);
  virtual const char* name();
  void backprop(const Vec& targetVals);
  void update_gradient(const Vec& in);
  void init_weights();
  void refine_weights(double learning_rate);
  void train_incremental(const Vec& feat, const Vec& lab);
  void train_unsupervised(const Matrix& X);
  void train_with_images(const Matrix& X);

  void setTestingData(const Matrix& testFeats, const Matrix& testLabs);

  float root_mean_squared_error(Matrix& features, Matrix& labels);
  void train(Matrix& trainFeats, Matrix& trainLabs, size_t batch_size, double momentum, double learning_rate);
  size_t countMisclassifications(const Matrix& features, const Matrix& labels);

  void centralFiniteDifferencing(Vec& x, Vec& target);
  void setReg(size_t);
  Matrix& getV();


  size_t effective_batch_size(double momentum);
  void scale_gradient(double scale);
  void add(Layer* l);
  Vec get_weights();


};



#endif

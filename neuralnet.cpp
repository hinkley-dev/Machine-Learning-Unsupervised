#include "neuralnet.h"
#include "supervised.h"

NeuralNet::NeuralNet(Rand r) : random(r)
{

}

NeuralNet::~NeuralNet()
{
  for(size_t i = 0; i < layers.size(); i++)
    		delete(layers[i]);
}

const char* NeuralNet::name()
{
  return "neural network";
}

Vec NeuralNet::get_weights()
{
  return weights;
}

void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs)
{

}

void NeuralNet::train_unsupervised(const Matrix& X)
{
    size_t n = X.rows();
    size_t k = 1;
    Matrix V(n, k);
    V.fill(0.0);

    double learning_rate = 0.1;

    for(size_t j = 0; j < 10; j++)
    {
        for(size_t i = 0; i <100000000; i++)
        {
           size_t t = random.next(X.rows());

            Vec features = V[t];

            Vec label = X[t];
            Vec prediction = predict(features);
            // compute the error on the output units
            backprop(label); //backprop now takes care of inputBlame

            update_gradient(features);
            refine_weights(learning_rate);

              //update gradient/refine weights for input
              if(inputBlame.size() != V[t].size()) throw Ex("input blame erros");
              for(size_t l = 0; l < inputBlame.size(); ++l)
                V[t][l] = V[t][l] + learning_rate*inputBlame[l];

        }
        learning_rate *= 0.75;

    }
}

void NeuralNet::train_with_images(const Matrix& X)
{
    size_t width = 64;
    size_t height = 48;
    size_t channels = X.cols() / (width * height);
    cout << "channels: " << channels << endl;

    //Initialize the MLP to have channels output units (not X.cols() outputs).
    size_t n = X.rows();
    size_t k = 2;
    Matrix V(n, k);
    V.fill(0.0);
    double learning_rate = 0.1;

    for(size_t j = 0; j < 10; j++)
    {
        for(size_t i = 0; i < 10000000; i++)
        {
          //
          // cout << "weights: " << endl;
          // weights.print();
          // cout << endl;

            size_t t = random.next(X.rows());
            size_t p = random.next(width);
            size_t q = random.next(height);
            Vec features(2 + V[t].size());
            features[0] = (double)p/width;
            features[1] = (double)q/height;


            //Vec vOft(V[t]);
            for(size_t l = 2, h = 0; l < features.size(); ++l, ++h)
              features[l] = V[t][h];

            //cout << "feat: " << features.size() << endl;

            size_t s = channels * (width * q + p);
            //cout << "s: " << s << endl;
                       // VecWrapper label = the vector from X[t][s] to X[t][s + (channels - 1)]
            Vec label(X[t], s, s + (channels - 1) + 1); //+1 to make it inclusive

            //cout << "labs: " << label.size() << endl;

            Vec pred = predict(features);
            // cout << "j= " << j << " i= " << i << endl;
            // cout << "t = " << t << "   p = " << p << "  q = " << q << endl;
            // cout << "in= ";
            // features.print();
            // cout << endl;
            // cout << "target= ";
            // label.print();
            // cout << endl;
            // cout << "prediction: ";
            // pred.print();
            // cout << endl;





            // compute the error on the output units
            backprop(label); //backprop now takes care of inputBlame

            update_gradient(features);
            refine_weights(learning_rate);

              //update gradient/refine weights for input

              //2 to start because of the width and height inputs

              for(size_t l = 2; l < inputBlame.size(); ++l)
              {

                 V[t][l-2] = V[t][l-2] + learning_rate*inputBlame[l];

              }
              // cout << endl;
              //
              //
              // cout << "updated V[t]: ";
              // V[t].print();
              // cout << endl;
              // cout << endl;
              // cout << endl;


        }
        cout << j << " / " << 10 << endl;
        learning_rate *= 0.75;
    }
    m_V.copy(V);
}

Matrix& NeuralNet::getV() { return m_V; }




void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs, size_t batch_size, double momentum, double learning_rate)
{

  size_t trainingDataCount = trainFeats.rows();

  size_t *randomIndicies= new size_t[trainingDataCount];
  for(size_t j = 0; j < trainingDataCount; ++j)
    randomIndicies[j] = j;


   random_shuffle(&randomIndicies[0],&randomIndicies[trainingDataCount]);


  for(size_t j = 0; j < trainingDataCount; ++j)
  {
    size_t row = randomIndicies[j];
    train_incremental(trainFeats.row(row), trainLabs.row(row));

     if(j % batch_size == 0 && j > 0)
     {
       refine_weights(learning_rate);
       scale_gradient(momentum);
     }
   }

  delete[] randomIndicies;
}

size_t NeuralNet::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");
	size_t mis = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];
		size_t predVal = pred.indexOfMax();
    size_t labVal = lab.indexOfMax();
			if(predVal != labVal)
			{
				mis++;
			}

	}
	return mis;
}


float NeuralNet::root_mean_squared_error(Matrix& features, Matrix& labels)
{
  float rmse = 0.0;
  float sse = sum_squared_error(features, labels);
  rmse = sqrt(sse / (features.rows()));
  return rmse;
}

void NeuralNet::setTestingData(const Matrix& testFeats, const Matrix& testLabs)
{
  testingFeatures.copy(testFeats);
  testingLabels.copy(testLabs);
}








void NeuralNet::scale_gradient(double scale)
{
  gradient *= scale;
}

void NeuralNet::train_incremental(const Vec& feat, const Vec& lab)
{

  predict(feat);
  backprop(lab);
  update_gradient(feat);

}



const Vec& NeuralNet::predict(const Vec& in)
{

  Vec layerInputs(in);
  size_t startWeight = 0;
  for(size_t i = 0; i < layers.size(); ++i)
  {
    VecWrapper layerWeights(weights, startWeight, layers[i]->getWeightCount());
    layers[i]->activate(layerWeights, layerInputs);

    layerInputs.copy(layers[i]->getActivation());
    startWeight += layers[i]->getWeightCount();
   // cout << "Layer " << i << " activation: " << endl;
    // layers[i]->getActivation().print();
    // cout << endl;
  }

  return layers[layers.size() -1]->getActivation();
}

void NeuralNet::backprop(const Vec& targetVals)
{
  Vec finalActivation(layers[layers.size() -1]->getActivation());
  Vec initialBlame(finalActivation.size());


  for(size_t i = 0; i < initialBlame.size(); ++i)
  {
    initialBlame[i] = targetVals[i] - finalActivation[i];
  }


  layers[layers.size() -1]->setBlame(initialBlame);

  size_t startWeight = weights.size();
  Vec prevBlame(initialBlame);

  for(size_t i = layers.size() - 1; i > 0; --i)
  {

    //build the weights

    layers[i]->setBlame(prevBlame);
    startWeight -= layers[i]->getWeightCount();
    VecWrapper layerWeights(weights, startWeight,layers[i]->getWeightCount());

    prevBlame.resize(layers[i]->getInputCount());
    prevBlame.fill(0.0);
    layers[i]->backprop(layerWeights, prevBlame);


  }

  layers[0]->setBlame(prevBlame);
  // cout << "Blame on layer 0: ";
  // prevBlame.print();
  // cout << endl;
  //set blane on some inputs
  inputBlame.resize(layers[0]->getInputCount());
  inputBlame.fill(0.0);

  VecWrapper firstLayerWeights(weights, 0, startWeight);
  // cout << "Layer 0 weights:" << endl;
  // firstLayerWeights.print();
  // cout << endl;
  layers[0]->backprop(firstLayerWeights, inputBlame);

  // cout << "input blame: ";
  // inputBlame.print();
  // cout << endl;
}

void NeuralNet::update_gradient(const Vec& x)
{

  if(&x == nullptr) throw Ex("input to update gradient is null");
  if(x.size() == 0) throw Ex("input is not the right size");
  Vec in(x);
  size_t startGradient = 0;
  gradient.fill(0.0);
  for(size_t i = 0; i < layers.size(); ++i)
  {

    VecWrapper layerGradient(gradient, startGradient, layers[i]->getWeightCount());
    layers[i]->update_gradient(in, layerGradient);

    //copying over the gradient
    for(size_t j = startGradient, k = 0; k < layerGradient.size(); ++j, ++k)
      gradient[j] = layerGradient[k];

    in.copy(layers[i]->getActivation());
    startGradient += layerGradient.size();
  }



}

void NeuralNet::init_weights()
{
  cout << "w size: " << weights.size() << endl;
  for(size_t i = 0; i < weights.size(); ++i)
  {
    weights[i] = 0.03*random.normal();
    //weights[i] = 0.1;
  }
//   weights.copy({0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,
//     0, 0.003, 0.006, 0.009,
//   0.007, 0.01, 0.013, 0.016,
//   0.014, 0.017, 0.02, 0.023,
//   0.021, 0.024, 0.027, 0.03,
//   0.028, 0.031, 0.034, 0.037,
//   0.035, 0.038, 0.041, 0.044,
//   0.042, 0.045, 0.048, 0.051,
//   0.049, 0.052, 0.055, 0.058,
//   0.056, 0.059, 0.062, 0.065,
//   0.063, 0.066, 0.069, 0.072,
//   0.07, 0.073, 0.076, 0.079,
//   0.077, 0.08, 0.083, 0.086,
// 0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,
// 0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027, 0.03, 0.033,
// 0.007, 0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031, 0.034, 0.037, 0.04,
// 0.014, 0.017, 0.02, 0.023, 0.026, 0.029, 0.032, 0.035, 0.038, 0.041, 0.044, 0.047,
// 0.021, 0.024, 0.027, 0.03, 0.033, 0.036, 0.039, 0.042, 0.045, 0.048, 0.051, 0.054,
// 0.028, 0.031, 0.034, 0.037, 0.04, 0.043, 0.046, 0.049, 0.052, 0.055, 0.058, 0.061,
// 0.035, 0.038, 0.041, 0.044, 0.047, 0.05, 0.053, 0.056, 0.059, 0.062, 0.065, 0.068,
// 0.042, 0.045, 0.048, 0.051, 0.054, 0.057, 0.06, 0.063, 0.066, 0.069, 0.072, 0.075,
// 0.049, 0.052, 0.055, 0.058, 0.061, 0.064, 0.067, 0.07, 0.073, 0.076, 0.079, 0.082,
// 0.056, 0.059, 0.062, 0.065, 0.068, 0.071, 0.074, 0.077, 0.08, 0.083, 0.086, 0.089,
// 0.063, 0.066, 0.069, 0.072, 0.075, 0.078, 0.081, 0.084, 0.087, 0.09, 0.093, 0.096,
// 0.07, 0.073, 0.076, 0.079, 0.082, 0.085, 0.088, 0.091, 0.094, 0.097, 0.1, 0.103,
// 0.077, 0.08, 0.083, 0.086, 0.089, 0.092, 0.095, 0.098, 0.101, 0.104, 0.107, 0.11,
// 0,0.001,0.002,
// 0, 0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021, 0.024, 0.027, 0.03, 0.033,
// 0.007, 0.01, 0.013, 0.016, 0.019, 0.022, 0.025, 0.028, 0.031, 0.034, 0.037, 0.04,
// 0.014, 0.017, 0.02, 0.023, 0.026, 0.029, 0.032, 0.035, 0.038, 0.041, 0.044, 0.047});

//weights.copy({0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3});
cout << "w size: " << weights.size() << endl;
  gradient.fill(0.0);

}

void NeuralNet::setReg(size_t r)
{
  reg = r;
}

void NeuralNet::refine_weights(double learning_rate)
{
  weights.addScaled(learning_rate, gradient);

}

void NeuralNet::centralFiniteDifferencing(Vec& x, Vec& target)
{

  Vec gradientCFD(gradient.size());
  gradientCFD.fill(0.0);
  Vec cp(weights);
  double lr = 0.0003;
  for(size_t i = 0; i < weights.size(); ++i)
  {
    double orig = weights[i];
    weights[i] += lr;
    Vec positiveStep = predict(x);
    double pos = 0.0;
    for(size_t j = 0; j < positiveStep.size(); j++)
		{
			pos+= ((positiveStep[j] - target[j])*(positiveStep[j] - target[j]));
		}
    weights[i] = orig - lr;
    Vec negativeStep = predict(x);
    double neg = 0.0;
    for(size_t j = 0; j < positiveStep.size(); j++)
    {
     neg+= ((negativeStep[j] - target[j])*(negativeStep[j] - target[j]));
    }
    gradientCFD[i] = (neg - pos)/ (2*lr);
    weights[i] = orig;
    for(size_t j = 0; j < weights.size(); ++j)
      if(weights[j] != cp[j]) throw Ex("weights not set back correctly");
  }
  train_incremental(x, target);
  size_t count = 0;
  for(size_t i = 0; i < gradient.size(); ++i)
  {
    if((gradientCFD[i] - gradient[i]) / gradientCFD[i] > 0.005)
      count++;
  }
  // cout << "Gradtients with central finite differencing: " << endl;
  // gradientCFD.print();
  // cout << endl;
  // cout << "Gradtients with backprop: " << endl;
  // gradient.print();
  // cout << endl;
  //  cout << endl;
  //   cout << endl;
  // cout << "Amount of times when the difference between backprop and central finite differencing is more than 0.5%: " << count << " / " << gradient.size() << endl;
  // cout << "The 10 can be accounted but the fact that they are the biases and my finite differencing test does not treat them any differently." << endl;


}


void NeuralNet::add(Layer* l)
{

  layers.push_back(l);
  weights.resize(weights.size() + layers[layers.size()-1]->getWeightCount());
  gradient.resize(weights.size());
}

size_t NeuralNet::effective_batch_size(double momentum)
{
  return (size_t)(1/(1-momentum));
}

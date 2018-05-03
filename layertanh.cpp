#include "layertanh.h"

LayerTanh::LayerTanh(size_t inputs) :
 Layer(inputs)
{
  m_inputs = inputs;
  weightCount = 0;
}

LayerTanh::~LayerTanh()
{

}

void LayerTanh::activate(const Vec& weights,const Vec& x)
{
  for(size_t i = 0; i < activation.size(); i++)
	{
		if(x[i] >= 700.0)
			activation[i] = 1.0;
		else if(x[i] < -700.0)
			activation[i] = -1.0;
		else activation[i] = tanh(x[i]);
	}
 //  cout << endl;
 //  cout << endl;
 //  cout << "In LayerTanh:activate = ";
 // activation.print();
 // cout << endl;


}

void LayerTanh::backprop(const Vec& weights, Vec& prevBlame)
{

  for(size_t i = 0; i < activation.size(); ++i)
  {
    prevBlame[i] = blame[i] * (1.0 - (activation[i] * activation[i]));
  }
  //
  // cout << endl;
  // cout << endl;
  // cout << "In LayerTanh::backpro" << endl;
  // cout << "blame on this layer: ";
  // blame.print();
  // cout << endl;
  // cout << "computed blame on prev layer: ";
  // prevBlame.print();
  // cout << endl;


}



size_t LayerTanh::getInputCount()
{
  return m_inputs;
}

void LayerTanh::update_gradient(const Vec& x, Vec& gradient)
{
  // cout << endl;
  // cout << endl;
  // cout << "In LayerTanh::update_gradient" << endl;
  // cout << "input vector: ";
  // x.print();
  // cout << endl;
  // cout << "blame vector: ";
  // blame.print();
  // cout << endl;

}

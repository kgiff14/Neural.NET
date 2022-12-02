using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Neural.NET.AppLogic.Cost.Implementations;
using Neural.NET.AppLogic.Cost.Interfaces;
using Neural.NET.AppLogic.Utilities;
using Neural.NET.Models;
using Numpy;

namespace Neural.NET.AppLogic.Layers.Implementations
{
    public class Trainer
    {
        private readonly ParameterInitializers _parameterInitializer;
        private readonly DenseLayer _denseLayer;
        private readonly BinaryCrossEntropy _costFunction;

        public Trainer()
        {
            _parameterInitializer = new ParameterInitializers();
            _denseLayer = new DenseLayer();
            _costFunction = new BinaryCrossEntropy();
        }

        public void Fit(NDarray X, NDarray Y, int[] layers, double learningRate = 0.0075, int numIterations = 3000, bool printCost = false)
        {
            np.random.seed(1);
            List<NDarray> costs = new List<NDarray>();

            var parameters = _parameterInitializer.InitializeParameters(layers);

            for (var i = 0; i < numIterations; i++)
            {
                var AL = _denseLayer.LModelForward(X, parameters);
                var cost = _costFunction.ComputeCost(AL.Item1, Y);
                var grads = _denseLayer.LModelBackward(AL.Item1, Y, AL.Item2);
                parameters = _denseLayer.UpdateParameters(parameters, grads, learningRate);

                if(printCost && (i % 100 == 0 || i == numIterations - 1))
                {
                    Console.WriteLine($"Cost after iteration {i}: {np.squeeze(cost)}");
                }

                if (i % 100 == 0 || i == numIterations)
                {
                    costs.Add(cost);
                }
            }
        }
    }
}

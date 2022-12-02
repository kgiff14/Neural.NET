using System;
using System.Collections.Generic;
using Neural.NET.AppLogic.Activations.Implementations;
using Neural.NET.AppLogic.Activations.Interfaces;
using Neural.NET.Models.Enums;
using Numpy;

namespace Neural.NET.AppLogic.Layers.Implementations
{
    public class DenseLayer
    {
        private readonly IActivationsFactory _activationsFactory;
        private Tuple<List<NDarray>, NDarray> _cache;

        public DenseLayer()
        {
            _activationsFactory = new ActivationsFactory();
        }

        internal Tuple<NDarray, Tuple<List<NDarray>, NDarray>> LinearActivationForward(NDarray A_prev, NDarray W, NDarray b, ActivationsType activationsType)
        {
            _activationsFactory.CreateActivations(activationsType);

            var linearCache = _activationsFactory.Activation?.LinearActivation(A_prev, W, b);
            var activationCache = _activationsFactory.Activation?.ForwardActivationCalculation(linearCache?.Item1 ?? throw new Exception());

            _cache = new Tuple<List<NDarray>, NDarray>(linearCache.Item2, activationCache.Item2);

            return new Tuple<NDarray, Tuple<List<NDarray>, NDarray>>(activationCache.Item1, _cache);
        }

        internal Tuple<NDarray, List<Tuple<List<NDarray>, NDarray>>> LModelForward(NDarray X, Dictionary<string, NDarray> parameters)
        {
            var A = X;
            var L = parameters.Count;
            List<Tuple<List<NDarray>, NDarray>> caches = new List<Tuple<List<NDarray>, NDarray>>();

            for (int i = 0; i < L; i++)
            {
                var A_prev = A;
                var cache = LinearActivationForward(A_prev, parameters["W" + i], parameters["b" + i], ActivationsType.ReLU);

                caches.Add(cache.Item2);
            }

            var AL = LinearActivationForward(A, parameters["W" + L], parameters["b" + L], ActivationsType.Sigmoid);
            caches.Add(AL.Item2);

            return new Tuple<NDarray, List<Tuple<List<NDarray>, NDarray>>>(AL.Item1, caches);
        }

        internal Tuple<NDarray, NDarray, NDarray> LinearActivationBackward(NDarray dA, Tuple<List<NDarray>, NDarray> cache, ActivationsType activationsType)
        {
            var linearCache = cache.Item1;
            var activationCache = cache.Item2;

            _activationsFactory.CreateActivations(activationsType);

            var dZ = _activationsFactory.Activation?.BackwardActivationCalculation(dA, activationCache);
            var result = _activationsFactory.Activation?.LinearBackward(dZ, linearCache);

            return result;
        }

        internal Dictionary<string, NDarray> LModelBackward(NDarray AL, NDarray Y, List<Tuple<List<NDarray>, NDarray>> caches)
        {
            Dictionary<string, NDarray> grads = new Dictionary<string, NDarray>();
            var L = caches.Count;
            var m = AL.shape[1];
            Y = Y.reshape(AL.shape);

            var dAL = np.divide(Y, AL) - np.divide(1 - Y, 1 - AL);

            var currentCache = caches[L - 1];
            var temps = LinearActivationBackward(dAL, currentCache, ActivationsType.Sigmoid);

            grads.Add("dA" + Convert.ToString(L - 1), temps.Item1);
            grads.Add("dW" + Convert.ToString(L), temps.Item2);
            grads.Add("db" + Convert.ToString(L), temps.Item3);

            for (var i = 0; i < L; i++)
            {
                currentCache = caches[i];
                temps = LinearActivationBackward(grads["dA" + Convert.ToString(i+1)], currentCache, ActivationsType.ReLU);

                grads["dA" + Convert.ToString(i)] = temps.Item1;
                grads["dW" + Convert.ToString(i + 1)] = temps.Item2;
                grads["db" + Convert.ToString(i + 1)] = temps.Item3;
            }

            return grads;
        }

        internal Dictionary<string, NDarray> UpdateParameters(Dictionary<string, NDarray> param, Dictionary<string, NDarray> grads, double learningRate)
        {
            var parameters = param;
            var L = parameters.Count;

            for(var i = 0; i < L; i++)
            {
                parameters["W" + Convert.ToString(i + 1)] = parameters["W" + Convert.ToString(i + 1)] - learningRate * grads["dW" + Convert.ToString(i + 1)];
                parameters["b" + Convert.ToString(i + 1)] = parameters["b" + Convert.ToString(i + 1)] - learningRate * grads["db" + Convert.ToString(i + 1)];
            }

            return parameters;
        }
    }
}

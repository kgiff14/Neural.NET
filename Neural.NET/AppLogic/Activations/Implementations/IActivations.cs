using System;
using System.Collections.Generic;
using Numpy;

namespace Neural.NET.AppLogic.Activations.Implementations
{
    public interface IActivations
    {
        public Tuple<NDarray, NDarray> ForwardActivationCalculation(NDarray z);

        public NDarray BackwardActivationCalculation(NDarray z, NDarray cache);

        public Tuple<NDarray, List<NDarray>> LinearActivation(NDarray A, NDarray W, NDarray b)
        {
            var result = np.dot(W, A) + b;

            return new Tuple<NDarray, List<NDarray>>(result, new List<NDarray> { A, W, b });
        }

        public Tuple<NDarray, NDarray, NDarray> LinearBackward(NDarray dZ, List<NDarray> cache)
        {
            var A_prev = cache[0];
            var W = cache[1];
            var b = cache[2];
            var m = A_prev.shape[1];

            var dW = 1/m  * np.dot(dZ, A_prev.T);
            var db = 1 / m * np.sum(dZ, axis: -1, keepdims: true);
            var dA_prev = np.dot(W.T, dZ);

            return new Tuple<NDarray, NDarray, NDarray>(dA_prev, dW, db);
        }
    }
}

using System;
using Neural.NET.AppLogic.Activations.Implementations;
using Numpy;

namespace Neural.NET.AppLogic.Activations.Interfaces
{
    public class Sigmoid : IActivations
    {
        public Tuple<NDarray, NDarray> ForwardActivationCalculation(NDarray z)
        {
            var result = SigmoidActivation(z);

            return new Tuple<NDarray, NDarray>(result, z);
        }

        public NDarray BackwardActivationCalculation(NDarray dA, NDarray cache)
        {
            var result = ForwardActivationCalculation(cache);
            var dZ = dA * result.Item1 * result.Item2 * (1-result.Item2);

            return dZ;
        }

        internal NDarray SigmoidActivation(NDarray z)
        {
            var result = 1 / (1 + np.exp(-z));

            return result;
        }
    }
}

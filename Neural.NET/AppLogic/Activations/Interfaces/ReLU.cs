using System;
using Neural.NET.AppLogic.Activations.Implementations;
using Numpy;

namespace Neural.NET.AppLogic.Activations.Interfaces
{
    public class ReLU : IActivations
    {
        public Tuple<NDarray, NDarray> ForwardActivationCalculation(NDarray z)
        {
            var result = ReluActivation(z);

            return new Tuple<NDarray, NDarray>(result, z);
        }

        public NDarray BackwardActivationCalculation(NDarray z, NDarray cache)
        {
            var result = ForwardActivationCalculation(cache);
            var dZ = z * (1 - np.square(result.Item1));

            return dZ;
        }

        internal NDarray ReluActivation(NDarray z)
        {
            var result = np.max(z, new int[0]);

            return result;
        }
    }
}

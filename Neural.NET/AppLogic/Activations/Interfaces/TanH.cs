using System;
using System.Collections.Generic;
using System.Text;
using Neural.NET.AppLogic.Activations.Implementations;
using Numpy;

namespace Neural.NET.AppLogic.Activations.Interfaces
{
    public class TanH : IActivations
    {
        public NDarray BackwardActivationCalculation(NDarray z)
        {
            var backPropResult = 1 - (TanhActivation(z) * TanhActivation(z));

            return backPropResult;
        }

        public NDarray BackwardActivationCalculation(NDarray z, NDarray cache)
        {
            throw new NotImplementedException();
        }

        public NDarray ForwardActivationCalculation(NDarray z)
        {
            var result = TanhActivation(z);

            return result;
        }

        internal NDarray TanhActivation(NDarray z)
        {
            var result = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z));

            return result;
        }

        Tuple<NDarray, NDarray> IActivations.ForwardActivationCalculation(NDarray z)
        {
            throw new NotImplementedException();
        }
    }
}

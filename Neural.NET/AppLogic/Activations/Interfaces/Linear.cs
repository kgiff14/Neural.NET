using System;
using System.Collections.Generic;
using Neural.NET.AppLogic.Activations.Implementations;
using Numpy;

namespace Neural.NET.AppLogic.Activations.Interfaces
{
    public class Linear
    {

        internal NDarray LinearActivation(NDarray A, NDarray W, NDarray b)
        {
            var result = np.dot(W,A) + b;

            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace Neural.NET.AppLogic.Cost.Implementations
{
    internal class BinaryCrossEntropy
    {
        public NDarray ComputeCost(NDarray AL, NDarray Y)
        {
            var m = Y.shape[1];

            var cost = -(1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)));
            cost = np.squeeze(cost);

            return cost;
        }
    }
}

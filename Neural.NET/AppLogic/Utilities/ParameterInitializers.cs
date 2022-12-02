using System;
using System.Collections.Generic;
using System.Text;
using Neural.NET.Models;
using Numpy;

namespace Neural.NET.AppLogic.Utilities
{
    internal class ParameterInitializers
    {
        internal Dictionary<string, NDarray> _parameters;

        internal Dictionary<string, NDarray> InitializeParameters(int[] layerDims)
        {
            np.random.seed(3);
            _parameters = new Dictionary<string, NDarray>();
            var L = layerDims.Length;

            for (int i = 1; i <= L; i++)
            {
                _parameters.Add("W" + Convert.ToString(i), np.random.randn(layerDims[i], layerDims[i - 1]) * 0.01);
                _parameters.Add("b" + Convert.ToString(i), np.zeros(layerDims[i], 1) * 0.01);
            }

            return _parameters;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;
using Neural.NET.AppLogic.Activations.Implementations;
using Neural.NET.Models.Enums;

namespace Neural.NET.AppLogic.Activations.Interfaces
{
    public class ActivationsFactory : IActivationsFactory
    {
        public IActivations? Activation { get; set; }

        public void CreateActivations(ActivationsType activationsType)
        {
            Activation = activationsType switch
            {
                //ActivationsType.Linear => new Linear(),
                ActivationsType.Sigmoid => new Sigmoid(),
                ActivationsType.ReLU => new ReLU(),
                ActivationsType.Tanh => new TanH(),
                _ => throw new ArgumentOutOfRangeException($"{activationsType} is not valid."),
            };
        }

    }
}

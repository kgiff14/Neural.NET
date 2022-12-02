using Neural.NET.Models.Enums;

namespace Neural.NET.AppLogic.Activations.Implementations
{
    public interface IActivationsFactory
    {
        IActivations? Activation { get; set; }

        public void CreateActivations(ActivationsType activationsType);
    }
}

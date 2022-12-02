using Neural.NET.Models.Enums;

namespace Neural.NET.Models
{
    public class Layer
    {
        public string? Name { get; set; }

        public ActivationsType ActivationsType { get; set; }

        public OptimizerType OptimizerType { get; set; }

        public CostType CostType { get; set; }

        public bool IsVectorized { get; set; }

        public Layer(ActivationsType activationsType = ActivationsType.ReLU, OptimizerType optimizerType = OptimizerType.Adam, CostType costType = CostType.MeanSquaredError, string? name = "", bool isVectorized = true)
        {
            Name = name;
            ActivationsType = activationsType;
            OptimizerType = optimizerType;
            CostType = costType;
            IsVectorized = isVectorized;
        }
    }
}

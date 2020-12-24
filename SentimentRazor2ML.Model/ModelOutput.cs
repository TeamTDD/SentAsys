using Microsoft.ML.Data;

namespace SentimentRazor2ML.Model
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
        public float[] Score { get; set; }
    }
}

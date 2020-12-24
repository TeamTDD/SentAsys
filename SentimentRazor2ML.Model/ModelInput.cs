using Microsoft.ML.Data;

namespace SentimentRazor2ML.Model
{
    public class ModelInput
    {
        [ColumnName("comment"), LoadColumn(0)]
        public string Comment{ get; set; }


        [ColumnName("toxic"), LoadColumn(1)]
        public string Toxic { get; set; }


    }
}

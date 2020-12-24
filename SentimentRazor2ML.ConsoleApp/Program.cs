using System;
using SentimentRazor2ML.Model;

namespace SentimentRazor2ML.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Comment = @"Верблюдов-то за что? Дебилы, бл...
",
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Использование модели для одного прогноза - Сравнение фактического негативного комментария с прогнозируемыми негативным комментарием из выборочных данных ...\n\n");
            Console.WriteLine($"Комментарий: {sampleData.Comment}");
            Console.WriteLine($"\n\nПредсказанное негативное значение: {predictionResult.Prediction} \nПредсказанная оценка: [{String.Join(",", predictionResult.Score)}]\n\n");
            Console.WriteLine("=============== Процесс завершен ===============");
            Console.ReadKey();
        }
    }
}

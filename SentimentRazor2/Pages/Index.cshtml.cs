using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.ML;
using SentimentRazor2ML.Model;

namespace SentimentRazor2.Pages
{
    public class IndexModel : PageModel
    {
        private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;
        public IndexModel(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }
        public void OnGet()
        {

        }

        public IActionResult OnGetAnalyzeSentiment([FromQuery] string text)
        {
            if (String.IsNullOrEmpty(text)) return Content("Neutral");
            var input = new ModelInput { Comment = text };
            //var inputPrepare = StringExtensions.ToBoolean(input.Toxic);
            var prediction = _predictionEnginePool.Predict(input);
            bool b = prediction.Prediction == "1";
            var toxic = Convert.ToBoolean(b) ? "Toxic" : "Not Toxic";
            return Content(toxic);
        }
    }
}

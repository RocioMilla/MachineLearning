using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
    //Creamos la clase training data, esto es lo que nuestra app va a aprender 
    class FeedbackTrainingData {
        [ColumnName(name: "Label")]
        public bool isGood { get; set; }
        public string feedbackText { get; set; }

    }
    class FeedbackPrediction {
        [ColumnName("PredictedLabel")]
        public bool isGood { get; set; }
    }
    class Program
    {
        static List<FeedbackTrainingData> trainingData = new List<FeedbackTrainingData>();
        // el test data va a ser para ver cuanto accuracy tiene (porcentaje de exactitud)
        static List<FeedbackTrainingData> testData = new List<FeedbackTrainingData>();

        // Función que carga los datos, este caso cargué unos 15.
        // Para que esto funcione correctamente debe tener varios datos donde aprender
        // de otra manera, el algoritmo puede fallar
        static void loadTestData()
        {
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "good",
                isGood = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "horrible terrible",
                isGood = false
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "nice",
                isGood = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "shit",
                isGood = false
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "sweet",
                isGood = true
            });
            testData.Add(new FeedbackTrainingData()
            {
                feedbackText = "average",
                isGood = true
            });
        }
        static void loadTrainigData()
        {
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "this is good",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "this is horrible",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "it very Average",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "bad horrible",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "well ok ok ",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "shitty terrible",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "soooo nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "cool nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "sweet and nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "nice and good",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "very good",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "quiet average",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "soooo nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "god horrible",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "average and ok",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "bad and hell",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "this nice but better can be done",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "bad bad",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "till now it looks nice",
                isGood = true
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "shit",
                isGood = false
            });
            trainingData.Add(new FeedbackTrainingData()
            {
                feedbackText = "oh this is shit",
                isGood = false
            });
        }
    
        static void Main(string[] args)
        {
            // cargamos los datos
            loadTrainigData();
            //Clase de ML.NET, la necesitamos para acceder a las operaciones del machine learning
            var mlContext = new MLContext();
            //Convertimos nuestros datos de trainingData en IDataView (Interfaz de ML.NET)
            IDataView dataView = mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(trainingData);
            // Creamos el pipeline
            //definimos nuestro algoritmo
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "feedbackText").Append(mlContext.BinaryClassification.Trainers.FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));
            // Entrenamos el algoritmo
            var model = pipeline.Fit(dataView);
            //Test data para ver accuracy
            loadTestData();
            IDataView dataViewTest = mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(testData);
            var predictions = model.Transform(dataViewTest);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);

            //
            Console.WriteLine("Feedback:");
            string feedbackString = Console.ReadLine();
            var predictionFunction = mlContext.Model.CreatePredictionEngine<FeedbackTrainingData, FeedbackPrediction>(model);
            var feedbackInput = new FeedbackTrainingData();
            feedbackInput.feedbackText = feedbackString;
            var feedbackPredicted = predictionFunction.Predict(feedbackInput);
            Console.WriteLine("Se predice:" + feedbackPredicted.isGood);
            Console.Read();
        }
    }
}

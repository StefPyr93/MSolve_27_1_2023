using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
    // for each double[] => rows: input quantity, columns: input dimension
    public interface INeuralNetwork
    {
        IEnumerable<NeuralNetworkLayerParameter> NeuralNetworkLayerParameters { get; }

        void Train(double[,] stimuli, double[,] responses);
        double[,] EvaluateResponses(double[,] data);
        double[][,] EvaluateResponseGradients(double[,] stimuli);

        void LoadNetwork(string netPath, string weightsPath, string normalizationXPath, string normalizationYPath);
        void SaveNetwork(string netPath, string weightsPath, string normalizationXPath, string normalizationYPath);
    }
}

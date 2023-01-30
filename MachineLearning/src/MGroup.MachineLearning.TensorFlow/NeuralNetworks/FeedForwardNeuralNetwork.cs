using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Losses;
using System.IO;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.KerasLayers;

namespace MGroup.MachineLearning.TensorFlow.NeuralNetworks
{
	public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        //public INetworkLayer[] neuralNetworkLayer { get; private set; }
        private Keras.Model model;
        private NDArray trainX, testX, trainY, testY;

        public int? Seed { get; }
        public int BatchSize { get; }
        public int Epochs { get; }
        public INetworkLayer[] NeuralNetworkLayer { get; private set; }
		public INormalization NormalizationX { get; private set; }
        public INormalization NormalizationY { get; private set; }
        public OptimizerV2 Optimizer { get; }
        public ILossFunc LossFunction { get; }
        public Layer[] Layer { get; private set; }

        public FeedForwardNeuralNetwork(INormalization normalizationX, INormalization normalizationY, OptimizerV2 optimizer, ILossFunc lossFunc, INetworkLayer[] neuralNetworkLayer, int epochs, int batchSize = -1, int? seed = 1)
        {
            BatchSize = batchSize;
            Epochs = epochs;
            Seed = seed;
            NormalizationX = normalizationX;
            NormalizationY = normalizationY;
            Optimizer = optimizer;
            LossFunction = lossFunc;
            NeuralNetworkLayer = neuralNetworkLayer;

            if (seed != null)
            {
                tf.set_random_seed(seed.Value);
            }
        }

		/// <summary>
		/// This constructor can be used for objects that will load their properties from external files.
		/// </summary>
		public FeedForwardNeuralNetwork()
		{
		}

		public void Train(double[,] stimuli, double[,] responses) => Train(stimuli, responses, null, null);

        public void Train(double[,] trainX, double[,] trainY, double[,] testX = null, double[,] testY = null)
        {
            tf.enable_eager_execution();

			PrepareData(trainX, trainY, testX, testY);

			CreateModel();

			model.compile(loss: LossFunction, optimizer: Optimizer, metrics: new[] { "accuracy" });
            model.fit(this.trainX, this.trainY, batch_size: BatchSize, epochs: Epochs, shuffle: false);

            if (testX != null && testY != null)
            {
                model.evaluate(this.testX, this.testY, batch_size: BatchSize);
            }
        }

		public double[,] EvaluateResponses(double[,] stimuli)
        {
            stimuli = NormalizationX.Normalize(stimuli);

            var npData = np.array(stimuli);
            var resultFull = model.Apply(npData, training: false);
            var resultSqueezed = tf.squeeze(resultFull).ToArray<double>();
            var responses = new double[stimuli.GetLength(0), resultFull.shape.dims[1]]; //.GetShape().as_int_list()[1]];
            for (int i = 0; i < responses.GetLength(0); i++)
            {
                for (int j = 0; j < responses.GetLength(1); j++)
                {
                    responses[i, j] = resultSqueezed[resultFull.shape.dims[1] * i + j];
                }
            }

            responses = NormalizationY.Denormalize(responses);

            return responses;
        }

        public double[][,] EvaluateResponseGradients(double[,] stimuli)
        {
            stimuli = NormalizationX.Normalize(stimuli);

            var responseGradients = new double[stimuli.GetLength(0)][,];
            for (int k = 0; k < stimuli.GetLength(0); k++)
            {
                var sample = new double[1, stimuli.GetLength(1)];
                for (int i = 0; i < stimuli.GetLength(1); i++)
                {
                    sample[0, i] = stimuli[k, i];
                }

                var ratioX = NormalizationX.ScalingRatio;

                var npSample = np.array(sample);
                using var tape = tf.GradientTape(persistent: true);
                {
                    tape.watch(npSample);
                    Tensor pred = model.Apply(npSample, training: false);
                    var ratioY = NormalizationY.ScalingRatio;

                    var numRowsGrad = pred.shape.dims[1];
                    var numColsGrad = npSample.GetShape().as_int_list()[1];
                    var slicedPred = new Tensor();
                    responseGradients[k] = new double[numRowsGrad, numColsGrad];
                    for (int i = 0; i < numRowsGrad; i++)
                    {
                        slicedPred = tf.slice<int, int>(pred, new int[] { 0, i }, new int[] { 1, 1 });
                        var slicedGrad = tape.gradient(slicedPred, npSample).ToArray<double>();
                        for (int j = 0; j < numColsGrad; j++)
                        {
                            responseGradients[k][i, j] = ratioY[i] / ratioX[j] * slicedGrad[j];
                        }
                    }
                }
            }

            return responseGradients;
        }

        public void SaveNetwork(string netPath, string weightsPath, string normalizationPath)
        {
            model.save_weights(weightsPath);

			using (Stream stream = File.Open(normalizationPath, false ? FileMode.Append : FileMode.Create))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				binaryFormatter.Serialize(stream, new INormalization[] { NormalizationX, NormalizationY });
			}

			using (Stream stream = File.Open(netPath, false ? FileMode.Append : FileMode.Create))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				binaryFormatter.Serialize(stream, NeuralNetworkLayer);
			}
		}

        public void LoadNetwork(string netPath, string weightsPath, string normalizationPath)
        {
			using (Stream stream = File.Open(normalizationPath, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				var Normalization = (INormalization[])binaryFormatter.Deserialize(stream);
				NormalizationX = Normalization[0];
				NormalizationY = Normalization[1];
			}

			using (Stream stream = File.Open(netPath, FileMode.Open))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				NeuralNetworkLayer = (INetworkLayer[])binaryFormatter.Deserialize(stream);
			}

			CreateModel();

			model.load_weights(weightsPath);
		}

		private void PrepareData(double[,] trainX, double[,] trainY, double[,] testX = null, double[,] testY = null)
		{
			if (testX != null)
			{
				var trainAndTestX = new double[trainX.GetLength(0) + testX.GetLength(0), trainX.GetLength(1)];
				trainX.CopyTo(trainAndTestX, 0);
				testX.CopyTo(trainAndTestX, trainX.Length);
				NormalizationX.Initialize(trainAndTestX, NormalizationDirection.PerColumn);
				//
				trainX = NormalizationX.Normalize(trainX);
				testX = NormalizationX.Normalize(testX);
			}
			else
			{
				NormalizationX.Initialize(trainX, NormalizationDirection.PerColumn);
				trainX = NormalizationX.Normalize(trainX);
			}

			if (testX != null)
			{
				var trainAndTestY = new double[trainY.GetLength(0) + testY.GetLength(0), trainY.GetLength(1)];
				trainY.CopyTo(trainAndTestY, 0);
				testY.CopyTo(trainAndTestY, trainY.Length);
				NormalizationY.Initialize(trainAndTestY, NormalizationDirection.PerColumn);
				//
				trainY = NormalizationY.Normalize(trainY);
				testY = NormalizationY.Normalize(testY);
			}
			else
			{
				NormalizationY.Initialize(trainY, NormalizationDirection.PerColumn);
				trainY = NormalizationY.Normalize(trainY);
			}

			this.trainX = np.array(trainX);
			this.trainY = np.array(trainY);

			if (testX != null && testY != null)
			{
				this.testX = np.array(testX, TF_DataType.TF_DOUBLE);
				this.testY = np.array(testY, TF_DataType.TF_DOUBLE);
			}
		}


		private void CreateModel()
		{
			keras.backend.clear_session();
			keras.backend.set_floatx(TF_DataType.TF_DOUBLE);
			if (!(NeuralNetworkLayer[0] is KerasLayers.InputLayer))
			{
				throw new NotImplementedException($"First layer must be of type IInputLayer");
			}

			var inputs = keras.Input(shape: ((KerasLayers.InputLayer)NeuralNetworkLayer[0]).InputShape[0], dtype: TF_DataType.TF_DOUBLE); //.as_int_list()[0]
			var outputs = inputs;

			for (int i = 1; i < NeuralNetworkLayer.Length; i++)
			{
				outputs = NeuralNetworkLayer[i].BuildLayer(outputs);
			}

			model = new Keras.Model(inputs, outputs, "current_model");

			model.summary();
		}
	}
}

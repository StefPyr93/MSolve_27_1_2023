using Tensorflow.NumPy;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using MGroup.MachineLearning.TensorFlow.Keras;
using Xunit;

namespace MGroup.MachineLearning.Tests
{

	public class Class4
	{
		[Fact]
		public static void Test1()
		{
			var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
			var y = np.array(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

			var inputs = keras.Input(shape: 2);
			var outputs = keras.layers.Dense(32, keras.activations.Relu).Apply(inputs);
			outputs = keras.layers.Dense(64, keras.activations.Relu).Apply(outputs);
			outputs = keras.layers.Dense(1, keras.activations.Sigmoid).Apply(outputs);
			var model = new Model(inputs, outputs, "current_model");
			model.compile(keras.optimizers.Adam(), keras.losses.MeanSquaredError(), new[] { "accuracy" });
			model.fit(x, y, epochs: 800, verbose: 2);
			var prediction = model.predict(x, 4);
		}

		[Fact]
		public static void Test2()
		{
			var x = np.array(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
			var y = np.array(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

			var model = keras.Sequential();
			model.add(keras.Input(2));
			model.add(keras.layers.Dense(32, keras.activations.Relu)); // 
			model.add(keras.layers.Dense(64, keras.activations.Relu));
			model.add(keras.layers.Dense(1, keras.activations.Sigmoid));
			model.compile(keras.optimizers.Adam(), keras.losses.MeanSquaredError(), new[] { "accuracy" });
			model.fit(x, y, epochs: 800, verbose: 2);
			var prediction = model.predict(x, 4);
		}
	}
}

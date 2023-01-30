using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Utils;
using System.IO;
using Tensorflow.Keras.Engine;
using Xunit;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using Xunit.Abstractions;
using Tensorflow.NumPy;
using Tensorflow.Keras.Losses;

namespace TensorFlowNET.Examples
{
	/// <summary>
	/// This tutorial shows how to classify images of flowers.
	/// https://www.tensorflow.org/tutorials/images/classification
	/// </summary>
	public class ImageClassificationKeras
	{
		int batch_size = 32;
		int epochs = 10;
		//Shape img_dim = (180, 180);
		//IDatasetV2 train_ds, val_ds;
		Model model;
		ILossFunc Loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);
		IDatasetV2 train_data;
		NDArray x_test, y_test, x_train, y_train;

		[Fact]
		public bool Run()
		{
			tf.enable_eager_execution();

			PrepareData();
			BuildModel();
			Train();

			return true;
		}

		public void BuildModel()
		{
			int num_classes = 10;
			//var normalization_layer = tf.keras.layers.Rescaling(1.0f / 255);
			//var layers = keras.layers;
			//model = keras.Sequential(new List<ILayer>
			//{
			//	layers.InputLayer(input_shape: (28, 28, 1)),
			//	layers.Rescaling(1.0f / 255),
			//	layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
			//	layers.MaxPooling2D(),
			//	layers.Flatten(),
			//	layers.Dropout(0.5f),
			//	layers.Dense(128, activation: keras.activations.Relu),
			//	layers.Dense(num_classes)
			//});

			var inputs = keras.Input(shape: (28, 28, 1));
			var outputs = new Rescaling(new RescalingArgs()
			{
				Scale = 1.0f / 255,
			}).Apply(inputs);
			//outputs = new Conv2D(new Conv2DArgs()
			//{
			//	Filters = 32,
			//	KernelSize = (5, 5),
			//	Activation = keras.activations.Relu,
			//}).Apply(outputs);
			//outputs = new MaxPooling2D(new MaxPooling2DArgs()
			//{
			//	PoolSize = (2, 2),
			//	Strides = 2,
			//}).Apply(outputs);
			//outputs = new Conv2D(new Conv2DArgs()
			//{
			//	Filters = 64,
			//	KernelSize = (3, 3),
			//	Activation = keras.activations.Relu,
			//}).Apply(outputs);
			//outputs = new MaxPooling2D(new MaxPooling2DArgs()
			//{
			//	PoolSize = (2, 2),
			//	Strides = 2,
			//}).Apply(outputs);
			outputs = new Flatten(new FlattenArgs()
			{
			}).Apply(outputs);
			outputs = new Dense(new DenseArgs()
			{
				Units = 1024,
				Activation = keras.activations.Relu,
			}).Apply(outputs);
			outputs = new Dropout(new DropoutArgs()
			{
				Rate = 0.5f,
			}).Apply(outputs);
			outputs = new Dense(new DenseArgs()
			{
				Units = 10,
				Activation = keras.activations.Softmax,
			}).Apply(outputs);

			model = new Functional(inputs, outputs, "current_model");

			model.compile(optimizer: keras.optimizers.Adam(),
				loss: Loss,
				metrics: new[] { "accuracy" });

			model.summary();
		}

		public void Train()
		{
			model.fit(train_data, batch_size: batch_size, epochs: epochs);

			var y_pred = model.Apply(x_test);
			var loss = Loss.Call(y_test, y_pred);
			var temp1 = tf.math.argmax(y_pred, 1);
			var temp2 = tf.cast(y_test, tf.int64);
			var correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), tf.cast(tf.squeeze(y_test), tf.int64));
			var accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
		}

		public void PrepareData()
		{
			((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
			x_train = x_train["::20"];
			y_train = y_train["::20"];
			x_test = x_test["::20"];
			y_test = y_test["::20"];
			x_train = tf.expand_dims(x_train).numpy();
			y_train = tf.expand_dims(y_train).numpy();
			x_test = tf.expand_dims(x_test).numpy();
			y_test = tf.expand_dims(y_test).numpy();
			//x_train = tf.convert_to_tensor(x_train, dtype: TF_DataType.TF_DOUBLE).numpy();
			//y_train = tf.convert_to_tensor(y_train, dtype: TF_DataType.TF_DOUBLE).numpy();
			//x_test = tf.convert_to_tensor(x_test, dtype: TF_DataType.TF_DOUBLE).numpy();
			//y_test = tf.convert_to_tensor(y_test, dtype: TF_DataType.TF_DOUBLE).numpy();
			train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
			train_data = train_data.repeat()
				.shuffle(5000)
				.batch(batch_size)
				.prefetch(1)
				.take(100);
		}
	}
}

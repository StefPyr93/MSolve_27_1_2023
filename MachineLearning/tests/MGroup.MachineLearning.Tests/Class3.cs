using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Utils;
using System.IO;
using Tensorflow.Keras.Engine;
using Xunit;
using Tensorflow.Keras.Losses;
using Tensorflow.NumPy;
using System.Linq;

namespace TensorFlowNET.Examples
{
	/// <summary>
	/// This tutorial shows how to classify images of flowers.
	/// https://www.tensorflow.org/tutorials/images/classification
	/// </summary>
	public class ImageClassificationKeras1
	{
		int batch_size = 32;
		int epochs = 1;
		Shape img_dim = (180, 180);
		ILossFunc Loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);
		IDatasetV2 train_ds, val_ds, train_ds_new;
		Model model;

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
			int num_classes = 5;
			// var normalization_layer = tf.keras.layers.Rescaling(1.0f / 255);
			var layers = keras.layers;
			model = keras.Sequential(new List<ILayer>
			{
				layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
				layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
				layers.MaxPooling2D(),
                /*layers.Conv2D(32, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),*/
                layers.Flatten(),
				layers.Dense(128, activation: keras.activations.Relu),
				layers.Dense(num_classes)
			});

			model.compile(optimizer: keras.optimizers.Adam(),
				loss: Loss,
				metrics: new[] { "accuracy" });

			model.summary();
		}

		public void Train()
		{
			model.fit(train_ds, validation_data: val_ds, epochs: epochs);
			var kek = train_ds.take(1);
			//var npImages = np.ndarray((32,180,180,3));
			//var npLabels = np.ndarray(32);
			foreach ((Tensor images,Tensor labels) in train_ds)
			{
				//npImages = (NDArray)npImages.Concat(images.numpy());
				var npImages = images.numpy();
				var npLabels = labels.numpy();
				var npPredLabels = model.Apply(npImages);
				var loss = Loss.Call(npLabels, npPredLabels);
			}
		}

		public void PrepareData()
		{
			string fileName = "flower_photos.tgz";
			string url = $"https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz";
			string data_dir = Path.Combine(Path.GetTempPath(), "flower_photos");
			Web.Download(url, data_dir, fileName);
			Compress.ExtractTGZ(Path.Join(data_dir, fileName), data_dir);
			data_dir = Path.Combine(data_dir, "flower_photos");

			// convert to tensor
			train_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
				validation_split: 0.2f,
				subset: "training",
				seed: 123,
				image_size: img_dim,
				batch_size: batch_size);

			val_ds = keras.preprocessing.image_dataset_from_directory(data_dir,
			validation_split: 0.2f,
			subset: "validation",
			seed: 123,
			image_size: img_dim,
			batch_size: batch_size);

			train_ds = train_ds.shuffle(1000).prefetch(buffer_size: -1);
			val_ds = val_ds.prefetch(buffer_size: -1);
		}
	}
}

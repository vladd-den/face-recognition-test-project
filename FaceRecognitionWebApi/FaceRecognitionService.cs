using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using System.Drawing;
using Emgu.CV.Util;

namespace FaceRecognitionWebApi
{
	public class FaceRecognitionService
	{
		private readonly CascadeClassifier _faceCascade;
		private readonly FaceRecognizer _faceRecognizer;

		public FaceRecognitionService()
		{
			// Load the pre-trained face cascade classifier
			_faceCascade = new CascadeClassifier("path/to/haarcascade_frontalface_default.xml");

			// Create the face recognizer
			_faceRecognizer = new LBPHFaceRecognizer();
			
			//comment this line if you dont have trained model, it will be created after training cycle
			_faceRecognizer.Read("trained_model.yml"); // Load the trained face recognition model
		}

		public List<string> DetectAndRecognizeFaces(string imagePath)
		{
			List<string> recognizedNames = new List<string>();

			// Load the image from file
			using (var image = new Image<Bgr, byte>(imagePath))
			{
				// Convert the image to grayscale
				using (var grayImage = image.Convert<Gray, byte>())
				{
					// Detect faces in the grayscale image
					var faces = _faceCascade.DetectMultiScale(grayImage, 1.1, 5, Size.Empty);

					// Process each detected face
					foreach (var face in faces)
					{
						// Crop the face region from the image
						var croppedFace = grayImage.GetSubRect(face);

						// Resize the cropped face to a fixed size (e.g., 100x100)
						var resizedFace = croppedFace.Resize(100, 100, Inter.Cubic);

						// Recognize the face
						var label = _faceRecognizer.Predict(resizedFace);

						// Add the recognized name to the list
						recognizedNames.Add(label.Label.ToString());
					}
				}
			}

			return recognizedNames;
		}

		public void Detect()
		{
			var faceRecognitionService = new FaceRecognitionService();

			// Provide the path to the image you want to process
			var imagePath = "path/to/image";

			// Detect and recognize faces in the image
			var recognizedNames = faceRecognitionService.DetectAndRecognizeFaces(imagePath);

			// Display the recognized names
			foreach (var name in recognizedNames)
			{
				Console.WriteLine("Recognized Name: " + name);
			}
		}

		public void TrainModel(Dictionary<string, List<Image<Gray, byte>>> labeledImages)
		{
			var faceRecognizer = new LBPHFaceRecognizer();

			// Convert the labeled images to the required format for training
			var faces = new List<Image<Gray, byte>>();
			var labels = new List<int>();

			foreach (var kvp in labeledImages)
			{
				var label = int.Parse(kvp.Key);

				foreach (var image in kvp.Value)
				{
					faces.Add(image);
					labels.Add(label);
				}
			}

			IInputArrayOfArrays facesInputArrayOfArrays = new VectorOfMat(faces.Select(img => img.Mat).ToArray());

			// Train the face recognition model
			faceRecognizer.Train(images: faces.Select(x => x.Mat).ToArray(), labels: labels.ToArray());

			// Save the trained model to a file
			faceRecognizer.Write("trained_model.yml");
		}

		public Dictionary<string, List<Image<Gray, byte>>> LoadTrainingImages(string directoryPath)
		{
			var labeledImages = new Dictionary<string, List<Image<Gray, byte>>>();

			// Get all the JPG files from the specified directory
			var jpgFiles = Directory.GetFiles(directoryPath, "*.jpg");

			// Process each JPG file
			foreach (var file in jpgFiles)
			{
				var fileName = Path.GetFileName(file);
				var label = fileName.Split('_')[0]; // Assuming the file names follow the format: "label_imageNumber.jpg for example 1_1.jpg"

				// Convert the image to grayscale
				var grayImage = new Image<Gray, byte>(file);

				// Add the grayscale image to the corresponding label
				if (labeledImages.ContainsKey(label))
				{
					labeledImages[label].Add(grayImage);
				}
				else
				{
					labeledImages[label] = new List<Image<Gray, byte>> { grayImage };
				}
			}

			return labeledImages;
		}
	}
}

using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace FaceRecognitionWebApi.Controllers
{
	[Route("face")]
	[ApiController]
	public class FaceController : ControllerBase
	{
		public FaceController()
		{

		}

		[HttpGet]
		public void Detect()
		{
			var faceRecognitionService = new FaceRecognitionService();

			faceRecognitionService.Detect();
		}

		[HttpPost]
		public void Train()
		{
			var faceRecognitionService = new FaceRecognitionService();

			// Specify the directory path where the training images are stored
			var trainingImagesDirectory = "path/to/training"; // training images should be in jpg format

			// Load the training images
			var labeledImages = faceRecognitionService.LoadTrainingImages(trainingImagesDirectory);

			// Train the model
			faceRecognitionService.TrainModel(labeledImages);
		}
	}
}

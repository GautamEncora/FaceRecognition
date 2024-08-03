const fileInput = document.getElementById('fileInput');
const submitBtn = document.getElementById('submitBtn');
const resultDiv = document.getElementById('result');
const submittedImage = document.getElementById('submittedImage');

const MODEL_URL = '/models'; // Path to your models directory

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
  faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
]).then((val) => {
  // console here gives an array of undefined
  console.log(val);
}).catch((err) => {
  console.log(err)
});

submitBtn.addEventListener('click', async () => {
  try {
    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
    const imageFile = fileInput.files[0];
    if (!imageFile) {
      alert("Please select an image file.");
      return;
    }

    const image = await faceapi.bufferToImage(imageFile);
    submittedImage.src = URL.createObjectURL(imageFile);
    submittedImage.style.display = 'block';

    const detections = await faceapi.detectSingleFace(image, new faceapi.SsdMobilenetv1Options()).withFaceLandmarks().withFaceDescriptor();
    
    if (!detections) {
      resultDiv.textContent = "No face detected. Please try another image.";
      return;
    }

    const bestMatch = faceMatcher.findBestMatch(detections.descriptor);
    const accuracy = (1 - bestMatch.distance) * 100;
    resultDiv.textContent = bestMatch.label === 'unknown' ? 
      "No match found." : 
      `Match found: ${bestMatch.label}, Accuracy: ${accuracy.toFixed(2)}%`;
  } catch (error) {
    console.error('Error processing image:', error);
    resultDiv.textContent = "An error occurred. Please try again.";
  }
});

async function loadLabeledImages() {
  const labels = ['sheldon','Amitabh','BillGates']; // Replace with your labels
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) { // Replace with the number of images per person
        const img = await faceapi.fetchImage(`/labeled_images/${label}/${i}.jpg`);
        const detections = await faceapi.detectSingleFace(img, new faceapi.SsdMobilenetv1Options()).withFaceLandmarks().withFaceDescriptor();
        if (detections) {
          descriptions.push(detections.descriptor);
        }
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

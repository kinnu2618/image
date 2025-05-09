let selectedImage = null;

document.getElementById("imageUpload").addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    selectedImage = file;

    const reader = new FileReader();
    reader.onload = function (event) {
      const imgElement = document.getElementById("preview");
      imgElement.src = event.target.result;
      imgElement.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});

function predict() {
  if (!selectedImage) {
    alert("Please upload an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedImage);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.json())
    .then((data) => {
      document.getElementById("result").textContent = "Prediction: " + data.prediction;
      document.getElementById("confidence").textContent = "Confidence: " + data.confidence.toFixed(4);
    })
    .catch((err) => {
      console.error("Error:", err);
      alert("Prediction failed.");
    });
}
